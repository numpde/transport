
# RA, 2018-11-11

from helpers import commons
from helpers import maps

import angles
import networkx as nx
import numpy as np
import math
import pickle
import random
import sklearn.neighbors
import geopy.distance
import gpxpy, gpxpy.gpx
from itertools import chain, groupby, product
from sklearn.neighbors import NearestNeighbors

# Metric for (lat, lon) coordinates
def geodist(a, b) :
	return geopy.distance.geodesic(a, b).m

# th is accuracy tolerance in meters
# TH is care-not radius for far-away segments
# Returns a pair (distance, lambda) where
# 0 <= lambda <= 1 is the relative location of the closest point
def dist_to_segment(x, ab, th=5, TH=1000) :
	# Endpoints of the segment
	(a, b) = ab

	# Relative location on the original segment
	(s, t) = (0, 1)

	# Distances to the endpoints
	(da, db) = (geodist(x, a), geodist(x, b))

	while True :

		# Relative accuracy of about 5% achieved?
		if (geodist(a, b) < 0.05 * min(da, db)) :
			break
			#(th < abs(da - db)) and (min(da, db) < TH)

		# Note: potentially problematic at lon~180
		# Approximation of the midpoint (m is not necessarily on a geodesic?)
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

# Signed "turning" angle (p, q) to (q, r) in degrees,
# where p, q, r are (lat, lon) coordinates
# https://stackoverflow.com/a/16180796/3609568
def angle(p, q, r) :
	# Note: the 'plus' is a concatenation of tuples here
	pq = angles.bear(*map(angles.d2r, p + q))
	qr = angles.bear(*map(angles.d2r, q + r))
	return (((angles.r2d(pq - qr) + 540) % 360) - 180)

def compute_knn(G : nx.DiGraph, locs, leaf_size=150):

	# Only care about the OSM nodes that are in the road network graph
	locs = { i: locs[i] for i in G.nodes() }

	(I, X) = (list(locs.keys()), list(locs.values()))

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

	# Road network (main graph component) with nearest-neighbor tree for the nodes
	G : nx.DiGraph
	knn_tree : NearestNeighbors
	(G, knn_ids, knn_tree) = commons.inspect(('g', 'node_ids', 'knn_tree'))(OSM['main_component_knn'])

	# Locations of the graph nodes
	node_pos = OSM['locs']

	# Free up some memory
	del OSM

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
	WP = list(map(commons.inspect({'StopPosition': ('PositionLat', 'PositionLon')}), motc_routes[route_id]['Stops'][direction]))

	#print(list(map(commons.inspect('StopName'), motc_routes['KHH122']['Stops'][0])))

	# DEBUG
	# WP = WP[0:3]
	# WP = WP[0:20]
	# WP = WP[-15:-11]

	#

	print("Locating nearest edges to waypoints...")

	def nearest_nodes(q, k=10) :

		# Find nearest nodes
		(dist, ind) = knn_tree.query(np.asarray(q).reshape(1, -1), k=k)

		# Get the in-graph node indices and flatten the arrays
		(dist, ind) = (dist.reshape(-1), [knn_ids[i] for i in ind.reshape(-1)])

		# Return a list of pairs (graph-node-id, distance-to-q) sorted by distance
		return list(zip(ind, dist))

	def nearest_edges(q, k=10) :
		# Get a number of closest nodes
		nn = [n for (n, d) in nearest_nodes(q, k=2*k+20)]
		# Get their incident edges
		ee = list(G.edges(nbunch=nn))
		# Append reverse edges
		ee = list(set(ee + [(b, a) for (a, b) in ee if G.has_edge(b, a)]))
		# Attach distance to q
		ee = [
			(e, dist_to_segment(q, (node_pos[e[0]], node_pos[e[1]]))[0])
			for e in ee
		]
		# Sort by distance
		ee = sorted(ee, key=(lambda ed : ed[1]))
		# Get the closest ones
		ee = ee[0:k]
		return ee

	# TODO: abort if nearest edges are too far

	# A cloud of edges for each waypoint
	dist_clouds = [dict(nearest_edges(wp)) for wp in WP]

	print("Connecting clouds...")

	# Edge clouds containing relative likelihood/suitability of edges for the optimal solution
	prob_clouds = [
		{
			# Initial likelihood inversely proportional to the distance to the waypoint
			e : (1 / (1 + d))
			for (e, d) in cloud.items()
		}
		for cloud in dist_clouds
	]

	# Intermediate edges -- random initial condition
	ee = [random.choices(list(pc.keys()), weights=list(pc.values()), k=1).pop() for pc in prob_clouds]

	# Remaining clouds to optimize (index into prob_clouds)
	rcto = list(range(len(ee)))

	# Quality markers
	(miss_dist, total_len) = (None, None)

	# Partial shortest path cache, keyed by tuples (start-node, end-node)
	sps_way = dict()


	# Plotting business

	import matplotlib.pyplot as plt

	fig : plt.Figure
	ax : plt.Axes
	(fig, ax) = plt.subplots()

	for (n, (y, x)) in enumerate(WP):
		ax.plot(x, y, 'bo')

	# Get the dimensions of the plot (again)
	(left, right, bottom, top) = ax.axis()

	# Compute a nicer aspect ratio if it is too narrow
	(w, h, phi) = (right - left, top - bottom, (1 + math.sqrt(5)) / 2)
	if (w < h / phi) : (left, right) = (((left + right) / 2 + s * h / phi / 2) for s in (-1, +1))
	if (h < w / phi) : (bottom, top) = (((bottom + top) / 2 + s * w / phi / 2) for s in (-1, +1))

	# Set new dimensions
	ax.axis([left, right, bottom, top])

	# Bounding box for the map
	bbox = (left, bottom, right, top)

	ax.autoscale(enable=False)

	# Download the background map
	mapi = maps.get_map_by_bbox(bbox, token=mapbox_token)

	plt.ion()
	plt.show()


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
	def opticrit(miss_dist, total_len):
		return crittime_from_meter_stop(miss_dist) + crittime_from_meter_bus(total_len)

	# Threshold turn for introducing "Detailed decision node clusters"
	ddnc_threshold = 60

	# Acceptance threshold for an edge in an edge cloud
	# (where certainty_level = 1 means complete certainty)
	certainty_level = 0.98

	# "Detailed decision node cluster" interface edge pairs (old, new)
	unmake_ddnc = []

	# Optimization loop
	while rcto :

		for _ in range(23) :
			if not rcto : break

			# Nodes to be replaced by "detailed decision node clusters"
			make_ddnc = set()

			try :

				# Choose a random edge cloud
				# nc = number of the edge cloud
				nc = random.choice(rcto)

				# Spend a few rounds on the same edge cloud
				for _ in range(10) :

					# pc = edge weights in this cloud (modified below)
					pc = prob_clouds[nc]

					# Cloud edges with weights
					(ec, cw) = map(list, (pc.keys(), pc.values()))
					# Exclude the currently selected edge
					(ec, cw) = zip(*[(e, p) for (e, p) in zip(ec, cw) if (e != ee[nc])])

					# Choose a candidate edge from the cloud (w/o the currently selected edge)
					ee[nc] = random.choices(ec, weights=cw, k=1).pop()

					# Precompute shortest path for each gap
					for ((_, a), (b, _)) in zip(ee, ee[1:]) :
						if (a, b) not in sps_way:
							sps_way[(a, b)] = nx.shortest_path(G, source=a, target=b, weight='len')

					# Previous metrics
					(old_miss_dist, old_total_len) = (miss_dist, total_len)

					# Criterion 1: Sum of distances of the selected edges to waypoints
					miss_dist = sum(dc[e] for (e, dc) in zip(ee, dist_clouds))

					# Criterion 2: The total length of the path
					path = [ee[0][0]] + list(chain.from_iterable(sps_way[(a, b)] for ((_, a), (b, _)) in zip(ee, ee[1:]))) + [ee[-1][-1]]
					total_len = sum(G.get_edge_data(*e)['len'] for e in zip(path, path[1:]))

					# Convert node IDs to (lat, lon) coordinates
					geo_path = [node_pos[i] for i in path]

					# Criterion 3: Turns
					for (p, q, r, b) in zip(geo_path, geo_path[1:], geo_path[2:], path[1:]) :
						# Skip nodes that are already part of a "detailed decision node cluster"
						if type(b) is tuple : continue
						# Upon a significant turn, schedule creation of a "detailed decision node cluster"
						if (abs(angle(p, q, r)) >= ddnc_threshold) : make_ddnc.add(b)

					#print("miss dist: {}, total len: {}".format(miss_dist, total_len))

					# Detailed decision node clusters scheduled
					if make_ddnc : break

					# Nothing to compare to in the first round
					if not (old_miss_dist and old_total_len) : continue

					# Consider this cloud "solved"?
					if (prob_clouds[nc][ee[nc]] >= certainty_level * sum(prob_clouds[nc].values())) :
						# Unschedule this cloud from further optimization
						rcto.remove(nc)
						break

					# Relative improvement new/old
					rel = opticrit(miss_dist, total_len) / opticrit(old_miss_dist, old_total_len)

					# Re-weight the current edge in its cloud
					prob_clouds[nc][ee[nc]] *= (2 if (rel < 1) else 0.5)

			except nx.NetworkXNoPath :
				raise


			# Replace selected nodes by "Detail decision node clusters"
			for nb in make_ddnc :

				# Incoming and outgoing edges
				iedges = list(G.in_edges(nbunch=nb))
				oedges = list(G.out_edges(nbunch=nb))

				# Make the cluster
				for (ie, oe) in product(iedges, oedges) :
					# New hyperedge between (new) hypernodes
					G.add_edge(ie, oe, len=meter_from_angle(angle(node_pos[ie[0]], node_pos[nb], node_pos[oe[1]])))
					# Geolocation of cluster hypernodes
					(node_pos[ie], node_pos[oe]) = [node_pos[nb]] * 2

				# Interface edges (old, new)
				e2E = dict( [(ie, (ie[0], ie)) for ie in iedges] + [(oe, (oe, oe[1])) for oe in oedges] )

				# Replace the interface edges, keeping the edge length
				for (e, E) in e2E.items() : G.add_edge(*E, len=G.get_edge_data(*e)['len'])

				# Remove old node and incident edges
				G.remove_node(nb)

				# To undo this process, record the interface edges
				unmake_ddnc.append(e2E)

				# Now fix the caches

				# Invalidated shortest paths
				sps_way = {ab: way for (ab, way) in sps_way.items() if not (nb in way)}

				for (e, E) in e2E.items():

					# Currently selected edges: replace e by E
					ee = [{e: E}.get(f, f) for f in ee]

					# Edge clouds: replace key e -> E
					for (nc, _) in enumerate(zip(prob_clouds, dist_clouds)) :
						try :
							dist_clouds[nc][E] = dist_clouds[nc].pop(e)
							prob_clouds[nc][E] = prob_clouds[nc].pop(e)
						except KeyError :
							pass


		# PLOT

		# Clear the axes
		ax.cla()

		# Apply the background map
		ax.axis((left, right, bottom, top))
		img = ax.imshow(mapi, extent=(left, right, bottom, top), interpolation='quadric', zorder=-100)

		if geo_path :
			(y, x) = zip(*geo_path)
			ax.plot(x, y, 'b--', linewidth=2, zorder=-50)

		for (n, (y, x)) in enumerate(WP) :
			c = 'm'
			ax.plot(x, y, 'o', c=c, markersize=4)

		for (nc, pc) in enumerate(prob_clouds) :
			m = max(pc.values())
			for (e, p) in pc.items() :
				(y, x) = zip(*[node_pos[i] for i in e])
				c = ('g' if (ee[nc] == e) else 'r')
				ax.plot(x, y, '-', c=c, linewidth=1, alpha=p/m, zorder=150)

		plt.draw()

		#plt.show()
		plt.pause(0.5)

	print("Finished")

	plt.ioff()
	plt.show()

	# # Note: this invalidates certain caches
	# while unmake_ddnc :
	# 	n = unmake_ddnc.pop()
	# 	G.remove_node(n)
	# 	G.add_edge(n)

	# GPX

	try :

		# Omit consecutive duplicates
		# https://stackoverflow.com/a/5738933
		geo_path = [next(iter(a)) for a in groupby(geo_path)]

		gpx = gpxpy.gpx.GPX()

		for (lat, lon) in WP :
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
