
# RA, 2018-11-11

from helpers import commons
from helpers import maps

import datetime as dt
import angles
import networkx as nx
import numpy as np
import math
import heapq
import pickle
import random
import sklearn.neighbors
import geopy.distance
import gpxpy, gpxpy.gpx
from itertools import chain, groupby, product
from sklearn.neighbors import NearestNeighbors

def geodist(a, b) :
	""" Metric for (lat, lon) coordinates """
	return geopy.distance.geodesic(a, b).m


# Signed "turning" angle (p, q) to (q, r) in degrees,
# where p, q, r are (lat, lon) coordinates
# https://stackoverflow.com/a/16180796/3609568
def angle(p, q, r) :
	# Note: the +plus+ is a concatenation of tuples here
	pq = angles.bear(*map(angles.d2r, p + q))
	qr = angles.bear(*map(angles.d2r, q + r))
	return (((angles.r2d(pq - qr) + 540) % 360) - 180)


def dist_to_segment(x, ab, rel_acc=0.05) :
	""" Compute the geo-distance of a point to a segment.
	
	Args:
		x: geo-coordinates of the point as a pair (lat, lon).
		ab: a pair (a, b) where a and b are endpoints of the segment.
		rel_acc: approximate relative accuracy (default 0.05).

	Returns:
		A pair (distance, t) with the approximate distance
			and a coordinate 0 <= t <= 1 of the closest point
			where t = 0 and t = 1 correspond to a and b, respectively.
	"""

	# Endpoints of the segment
	(a, b) = ab

	# Relative location on the original segment
	(s, t) = (0, 1)

	# Distances to the endpoints
	(da, db) = (geodist(x, a), geodist(x, b))

	while (geodist(a, b) > rel_acc * min(da, db)) :

		# Note: problematic at lon~180
		# Approximation of the midpoint (m is not necessarily on a geodesic)
		m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

		# Distance to the midpoint
		dm = geodist(x, m)

		if (da < db) :
			# Keep bisecting the (a, m) segment
			(b, db, t) = (m, dm, (s + t) / 2)
		else :
			# Keep bisecting the (m, b) segment
			(a, da, s) = (m, dm, (s + t) / 2)

	return min((da, s), (db, t))


def mapmatch(waypoints: list, g: nx.DiGraph, knn: callable, callback: callable = None) :
	""" Find a plausible bus route that visits the waypoints.

	Returns:
		A dictionary with the following keys
			- 'path' is a list of nodes of the graph g, as an estimate of the route
			- 'geo_path' is a list of (lat, lon) coordinates of those nodes
			- 'edge_clouds' is a list of edge clouds, one for each waypoint
			- 'active_edges' is a list of currently selected edges, one for each edge cloud
	"""

	# A turn of 90 degrees takes on average 25 seconds (wild guess)
	# Assume an average bus speed of 6 m/s to compute equivalent distance in meters
	# Penalize U-turns equivalently to a 200m run
	# Note: could distinguish left and right turns here
	meter_from_angle = (lambda a : (((25 * abs(a) / 90) * 6) if (abs(a) < 150) else 200))
	# Assume an average bus speed of 6 m/s
	crittime_from_meter_bus = (lambda d : (d / 6))
	# Convert the distance busstop-road to bus-time equivalent
	crittime_from_meter_stop = (lambda d : (5 * d / 1.5))

	# Optimization criterion
	def opticrit(indi):
		return crittime_from_meter_stop(indi['miss']) + crittime_from_meter_bus(indi['dist'])

	# Threshold turn for introducing "Detailed decision node clusters"
	ddnc_threshold = 60

	# Acceptance threshold for an edge in an edge cloud
	# (where 1 means "complete")
	acceptance_threshold = 0.98

	# Copies of objects that will/could get modified
	g = g

	# Check for the existence of those attributes
	nx.get_node_attributes(g, 'pos') # Node (lat, lon)
	nx.get_edge_attributes(g, 'len') # Edge lengths

	# Mark the original nodes as basenodes
	for n in g.nodes : g.nodes[n]['basenode'] = n

	# Locate k directed edges nearest to a point q
	def nearest_edges(q, k=10) :
		# Get some closest nodes first
		nn = list(dict(knn(q, k=2*(k+5))).keys())
		# Get their incident edges
		ee = g.edges(nbunch=nn)
		# Append reverse edges
		ee = set(list(ee) + [(b, a) for (a, b) in ee if g.has_edge(b, a)])
		# Attach distance from q
		ee = [
			(e, dist_to_segment(q, (g.nodes[e[0]]['pos'], g.nodes[e[1]]['pos']))[0])
			for e in ee
		]
		# Get the closest edges
		ee = list(heapq.nsmallest(k, ee, key=(lambda ed : ed[1])))

		return ee

	# A cloud of candidate edges for each waypoint
	dist_clouds = [dict(nearest_edges(wp)) for wp in waypoints]

	# Edge clouds containing relative suitability of edges for the optimal solution
	prob_clouds = [
		{
			# Initial likelihood inversely proportional to the distance to the waypoint
			e : (1 / (1 + d))
			for (e, d) in dc.items()
		}
		for dc in dist_clouds
	]

	# Intermediate edges -- random initial condition
	seledges = [random.choices(list(pc.keys()), weights=list(pc.values()), k=1).pop() for pc in prob_clouds]

	# Partial shortest path cache, keyed by tuples (start-node, end-node)
	sps_way = dict()

	# Route quality indicators
	indi = dict()

	# Dictionary to pass to the callback function
	result = { 'status' : 'init' }

	# Report current status
	if callback : callback(result)

	# Replace node by "Detailed decision node clusters"
	def make_ddnc(nb) :

		# Incoming and outgoing edges
		iedges = list(g.in_edges(nbunch=nb))
		oedges = list(g.out_edges(nbunch=nb))

		# Make the cluster
		for (ie, oe) in product(iedges, oedges):
			# New hyperedge between (new) hypernodes
			g.add_edge(ie, oe, len=meter_from_angle(angle(*(g.nodes[n]['pos'] for n in (ie[0], nb, oe[1])))))

		# Interface edges (old, new)
		e2E = dict([(ie, (ie[0], ie)) for ie in iedges] + [(oe, (oe, oe[1])) for oe in oedges])

		for (e, E) in e2E.items():
			# Refer new nodes to the basenode
			g.nodes[e]['basenode'] = g.nodes[nb]['basenode']
			# Geolocation of the hypernodes
			g.nodes[e]['pos'] = g.nodes[nb]['pos']
			# Replace the interface edges, keeping the edge length
			g.add_edge(*E, len=g.get_edge_data(*e)['len'])

		# Remove old node and incident edges
		g.remove_node(nb)

		# Now fix the caches

		# Remove invalidated shortest paths
		for (ab, way) in list(sps_way.items()) :
			if b in way :
				sps_way.pop(ab)

		for (e, E) in e2E.items():

			# Currently selected edges: replace e by E
			while e in seledges :
				seledges[seledges.index(e)] = E

			# Edge clouds: replace key e -> E
			for (nc, _) in enumerate(zip(prob_clouds, dist_clouds)):
				try:
					dist_clouds[nc][E] = dist_clouds[nc].pop(e)
					prob_clouds[nc][E] = prob_clouds[nc].pop(e)
				except KeyError:
					pass

	# Remaining edge clouds to optimize (index into prob_clouds)
	recto = list(range(len(seledges)))

	# Optimization loop
	while recto :

		# Choose a random edge cloud, preferably the "least solved" one
		# nc = number of the edge cloud
		nc = random.choices(recto, weights=[(lambda v : (1 - max(v) / sum(v)))(prob_clouds[nc].values()) for nc in recto], k=1).pop()

		#print(min(max(pc.values()) / sum(pc.values()) for pc in prob_clouds), recto)

		# Spend a few rounds on the same edge cloud
		while True :

			# Cloud edges with weights w/o the currently selected edge
			(ce, cw) = zip(*[(e, p) for (e, p) in prob_clouds[nc].items() if (e != seledges[nc])])

			# Choose a candidate edge from the cloud (w/o the currently selected edge)
			seledges[nc] = random.choices(ce, weights=cw, k=1).pop()

			# Precompute shortest path for each gap
			for ((_, a), (b, _)) in zip(seledges, seledges[1:]) :
				if (a, b) not in sps_way:
					try :
						sps_way[(a, b)] = nx.shortest_path(g, source=a, target=b, weight='len')
					except nx.NetworkXNoPath :
						raise

			path = [seledges[0][0]] + list(chain.from_iterable(sps_way[(a, b)] for ((_, a), (b, _)) in zip(seledges, seledges[1:]))) + [seledges[-1][-1]]

			# Convert node IDs to (lat, lon) coordinates
			geo_path = [g.nodes[n]['pos'] for n in path]

			# Undo "Detailed decision node cluster"s for the user
			origpath = [g.nodes[n]['basenode'] for n in path]


			# How certain we are about about the currently selected edge
			cloud_certainty = prob_clouds[nc][seledges[nc]] / sum(prob_clouds[nc].values())

			# Consider this cloud solved?
			if (cloud_certainty >= acceptance_threshold) :
				# Unschedule this cloud from further optimization
				recto.remove(nc)
				break

			# Introduct turn penalization
			if (cloud_certainty >= 0.3) :

				# Nodes to be replaced by "detailed decision node clusters"
				def get_cand_ddnc() :
					for (p, q, r, b) in zip(geo_path, geo_path[1:], geo_path[2:], path[1:]) :
						# Check if the node is original and the turn is significant
						if (g.nodes[b]['basenode'] == b) and (abs(angle(p, q, r)) >= ddnc_threshold) :
							yield b

				# Retain only the nodes influenced by the currently selected edge
				cand_ddnc = set(get_cand_ddnc()) & set(chain.from_iterable(
					sps_way[(e[1], f[0])]
					for (e, f) in zip(seledges, seledges[1:]) if (seledges[nc] in [e, f])
				))

				if cand_ddnc :
					# Introduce DDNC
					for b in cand_ddnc : make_ddnc(b)
					# Continue optimizing the current cloud w/o updating the indicators
					continue

			# Previous value of the quality indicators
			old_indi = indi.copy()

			# Indicator 1: Sum of distances of the selected edges to waypoints
			indi['miss'] = sum(dc[e] for (e, dc) in zip(seledges, dist_clouds))

			# Indicator 2: The total length of the path
			indi['dist'] = sum(g.get_edge_data(*e)['len'] for e in zip(path, path[1:]))

			# Re-weight the current edge in its cloud
			if not old_indi : continue
			prob_clouds[nc][seledges[nc]] *= (2 if (opticrit(indi) < opticrit(old_indi)) else 0.5)

			# Leave this edge cloud (more likely for nearly-solved clouds)
			if (random.random() < cloud_certainty) : break

		# Callback

		result['status'] = 'opti'
		result['path'] = origpath
		result['geo_path'] = geo_path
		result['edge_clouds'] = prob_clouds
		result['active_edges'] = seledges

		if callback: callback(result)

	result['status'] = 'done'
	if callback: callback(result)

	return result


def compute_knn(g: nx.DiGraph, node_pos, leaf_size=20):

	(I, X) = (list(node_pos.keys()), list(node_pos.values()))

	return {
		'node_ids': I,
		'knn_tree': sklearn.neighbors.BallTree(X, leaf_size=leaf_size, metric='pyfunc', func=commons.geodesic)
	}


def foo() :

	random.seed(0)

	mapbox_token = open("../.credentials/UV/mapbox-token.txt", 'r').read()

	osm_graph_file = "../OUTPUT/02/UV/kaohsiung.pkl"

	print("Loading OSM...")
	OSM = pickle.load(open(osm_graph_file, 'rb'))

	print("Retrieving the KNN tree...")

	# Road network (main graph component) with nearest-neighbor tree for the nodes
	g : nx.DiGraph
	knn_tree : NearestNeighbors
	(g, knn_ids, knn_tree) = commons.inspect(('g', 'node_ids', 'knn_tree'))(OSM['main_component_knn'])

	# Locations of the graph nodes
	nx.set_node_attributes(g, OSM['locs'], 'pos')

	# Free up memory
	del OSM

	#

	# Get some waypoints
	routes_file = "../OUTPUT/00/ORIGINAL_MOTC/Kaohsiung/CityBusApi_StopOfRoute.json"

	motc_routes = commons.index_dicts_by_key(
		commons.zipjson_load(routes_file),
		lambda r: r['RouteUID'],
		preserve_singletons=['Direction', 'Stops']
	)

	# Waypoints
	# (route_id, direction) = ('KHH1221', 0)
	# (route_id, direction) = ('KHH29', 0)
	# (route_id, direction) = ('KHH38', 0)
	# (route_id, direction) = ('KHH87', 1)
	# (route_id, direction) = ('KHH121', 1)
	(route_id, direction) = ('KHH11', 1)
	#
	waypoints = list(map(commons.inspect({'StopPosition': ('PositionLat', 'PositionLon')}), motc_routes[route_id]['Stops'][direction]))

	#

	def nearest_nodes(q, k=10) :

		# Find nearest nodes
		(dist, ind) = knn_tree.query(np.asarray(q).reshape(1, -1), k=k)

		# Get the graph node indices and flatten the arrays
		(dist, ind) = (dist.reshape(-1), [knn_ids[i] for i in ind.reshape(-1)])

		# Return pairs (graph-node-id, distance-to-q) sorted by distance
		return list(zip(ind, dist))

	# TODO: abort if nearest edges are too far

	try :
		# Plotting business

		import matplotlib.pyplot as plt

		fig : plt.Figure
		ax : plt.Axes
		(fig, ax) = plt.subplots()

		for (n, (y, x)) in enumerate(waypoints):
			ax.plot(x, y, 'bo')

		# Get the dimensions of the plot (again)
		(left, right, bottom, top) = ax.axis()

		# Compute a nicer aspect ratio if it is too narrow
		(w, h, phi) = (right - left, top - bottom, (1 + math.sqrt(5)) / 2)
		if (w < h / phi) : (left, right) = (((left + right) / 2 + s * h / phi / 2) for s in (-1, +1))
		if (h < w / phi) : (bottom, top) = (((bottom + top) / 2 + s * w / phi / 2) for s in (-1, +1))

		# Set new dimensions
		ax.axis([left, right, bottom, top])
		ax.autoscale(enable=False)

		# Download the background map
		mapi = maps.get_map_by_bbox((left, bottom, right, top), token=mapbox_token)

		plt.ion()
		plt.show()
	except :
		raise


	def mm_callback(result) :

		if (result['status'] == "init") :
			print("(Optimizing)")
			return

		if (result['status'] == "done") :
			print("(Done)")

		if (result['status'] == "opti") :
			if (dt.datetime.now() < result.get('nfu', dt.datetime.min)) :
				return

		# Clear the axes
		ax.cla()

		# Apply the background map
		ax.axis((left, right, bottom, top))
		img = ax.imshow(mapi, extent=(left, right, bottom, top), interpolation='none', zorder=-100)

		(y, x) = zip(*result['geo_path'])
		ax.plot(x, y, 'b--', linewidth=2, zorder=-50)

		for (n, (y, x)) in enumerate(waypoints) :
			c = 'm'
			ax.plot(x, y, 'o', c=c, markersize=4)

		for (nc, pc) in enumerate(result['edge_clouds']) :
			m = max(pc.values())
			for (e, p) in pc.items() :
				(y, x) = zip(*[g.nodes[i]['pos'] for i in e])
				c = ('g' if (result['active_edges'][nc] == e) else 'r')
				ax.plot(x, y, '-', c=c, linewidth=1, alpha=p/m, zorder=150)

		plt.pause(0.1)

		# Next figure update
		result['nfu'] = dt.datetime.now() + dt.timedelta(seconds=2)


	print("Connecting clouds...")

	result = mapmatch(waypoints, g, nearest_nodes, mm_callback)

	plt.ioff()
	plt.show()

	return

	# GPX

	try :

		# Omit consecutive duplicates
		# https://stackoverflow.com/a/5738933
		geo_path = [next(iter(a)) for a in groupby(geo_path)]

		gpx = gpxpy.gpx.GPX()

		for (lat, lon) in waypoints :
			gpx.waypoints.append(gpxpy.gpx.GPXWaypoint(latitude=lat, longitude=lon))

		gpx_track = gpxpy.gpx.GPXTrack()
		gpx.tracks.append(gpx_track)

		gpx_segment = gpxpy.gpx.GPXTrackSegment()
		gpx_track.segments.append(gpx_segment)

		for (p, q) in geo_path :
			gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=p, longitude=q))

		with open("tmp.gpx", 'w') as f:
			f.write(gpx.to_xml())
	except Exception as e :
		print("Writing GPX failed ({})".format(e))


if (__name__ == "__main__") :
	foo()
