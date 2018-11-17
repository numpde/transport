
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
from copy import deepcopy
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


def mapmatch(
		waypoints: list,
		g: nx.DiGraph,
		kne: callable,
		callback: callable = None,
		stubborn: float = 1
):
	""" Find a plausible bus route that visits the waypoints.

	Returns:
		A dictionary with the following keys
			- 'path' is a list of nodes of the graph g, as an estimate of the route
			- 'geo_path' is a list of (lat, lon) coordinates of those nodes
			- 'edge_clouds' is a list of edge clouds, one for each waypoint
			- 'active_edges' is a list of currently selected edges, one for each edge cloud
	"""

	# Dictionary to pass to the callback function (updated below)
	result = { 'waypoints' : waypoints, 'version' : 11111111 }

	# Before doing anything
	result['status'] = "zero"
	if callback : callback(result)

	# A turn of 90 degrees takes on average 25 seconds (wild guess)
	# Assume an average bus speed of 6 m/s to compute equivalent distance in meters
	# Penalize U-turns equivalently to a long run
	# Note: could distinguish left and right turns here
	meter_from_angle = (lambda a : (((25 * abs(a) / 90) * 6) if (abs(a) < 150) else 300))
	# Assume an average bus speed of 6 m/s
	crittime_from_meter_bus = (lambda d : (d / 6))
	# Convert the distance busstop-road to bus-time equivalent
	crittime_from_meter_stop = (lambda d : (5 * d / 1.5))

	# Optimization criterion (lower is better)
	def opticrit(indi):
		return stubborn * crittime_from_meter_stop(indi['miss']) + crittime_from_meter_bus(indi['dist'])

	# Threshold turn for introducing "Detailed decision node clusters"
	ddnc_threshold = 30

	# Acceptance threshold (between 0 and 1) for edge confidence in an edge cloud
	acceptance_threshold = 0.98

	# The graph will be modified
	g = deepcopy(g)

	# Check for the existence of those attributes
	assert(nx.get_node_attributes(g, 'pos')) # Node (lat, lon)
	assert(nx.get_edge_attributes(g, 'len')) # Edge lengths

	# Mark the original nodes as basenodes
	for n in g.nodes : g.nodes[n]['basenode'] = n

	# A cloud of candidate edges for each waypoint
	dist_clouds = [dict(kne(wp)) for wp in waypoints]

	# Edge clouds containing relative suitability of edges for the optimal solution
	prob_clouds = [
		{
			# Initial likelihood inv-prop to the regularized distance (in meters) to the waypoint
			e : (1 / (5 + d))
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

	# Intialized
	result['status'] = "init"
	if callback : callback(result)

	# Optimization loop
	for group_size in np.round(np.logspace(np.log10(2), np.log10(len(prob_clouds)), 5)) :
		# Get rid of np datatype
		group_size = int(group_size)

		group_acceptance_threshold = acceptance_threshold * (group_size / len(prob_clouds))

		for _ in np.arange(np.round(len(prob_clouds) / group_size * 1.4)) :

			# (Remaining) edge clouds to optimize (index into prob_clouds)
			# Random contiguous selection of length "group_size"
			recto = list(random.choice([range(n, n + group_size) for n in range(len(prob_clouds) - group_size + 1)]))

			while recto :

				# Normalize (not strictly necessary)
				prob_clouds = [{ e : (p / sum(pc.values())) for (e, p) in pc.items() } for pc in prob_clouds]

				# Choose a random edge cloud, preferably the "least solved" one
				# nc = number of the edge cloud
				nc = random.choices(recto, weights=[(lambda v : (2 - max(v) / sum(v)))(prob_clouds[nc].values()) for nc in recto], k=1).pop()

				#print(min(max(pc.values()) / sum(pc.values()) for pc in prob_clouds), recto)
				overall_progress = np.mean([min(1, max(pc.values()) / sum(pc.values()) / acceptance_threshold) for pc in prob_clouds])

				select_new_edge = True

				# Spend a few rounds on the same edge cloud
				while True :

					if select_new_edge :
						# Cloud edges with weights w/o the currently selected edge
						(ce, cw) = zip(*[(e, p) for (e, p) in prob_clouds[nc].items() if (e != seledges[nc])])

						# Choose a candidate edge from the cloud (w/o the currently selected edge)
						seledges[nc] = random.choices(ce, weights=cw, k=1).pop()
					else :
						select_new_edge = True

					# Precompute shortest path for each gap
					try :
						for ((_, a), (b, _)) in zip(seledges, seledges[1:]) :
							if (a, b) not in sps_way:
								sps_way[(a, b)] = nx.shortest_path(g, source=a, target=b, weight='len')
					except nx.NetworkXNoPath :
						raise

					path = [seledges[0][0]] + list(chain.from_iterable(sps_way[(a, b)] for ((_, a), (b, _)) in zip(seledges, seledges[1:]))) + [seledges[-1][-1]]

					# Convert node IDs to (lat, lon) coordinates
					geo_path = commons.remove_repeats([g.nodes[n]['pos'] for n in path])

					# Undo "Detailed decision node cluster"s for the user
					origpath = commons.remove_repeats([g.nodes[n]['basenode'] for n in path])

					# Previous value of the quality indicators
					old_indi = indi.copy()

					# Indicator 1: Sum of distances of the selected edges to waypoints
					indi['miss'] = sum(dc[e] for (e, dc) in zip(seledges, dist_clouds))

					# Indicator 2: The total length of the path
					indi['dist'] = sum(g.get_edge_data(*e)['len'] for e in zip(path, path[1:]))

					# Optimization criterion (lower is better)
					indi['crit'] = opticrit(indi)

					# Re-weight the current edge in its cloud
					if not old_indi : continue
					rel_delta = indi['crit'] / old_indi['crit']
					prob_clouds[nc][seledges[nc]] *= (1.3 if (rel_delta < 1) else (0.8 / rel_delta))


					# How certain we are about about the currently selected edge (again)
					cloud_certainty = prob_clouds[nc][seledges[nc]] / sum(prob_clouds[nc].values())

					# Consider this cloud solved?
					if (cloud_certainty >= group_acceptance_threshold) :
						# Unschedule this cloud from further optimization
						recto.remove(nc)
						break

					# Randomly leave this edge cloud (more likely for nearly-solved clouds)
					if (not (rel_delta > 1)) and (random.random() < cloud_certainty/2) :
						break

					# Introduce turn penalization
					if (cloud_certainty >= 0.2) :

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
							# The graph structure changed; invalidate metrics
							indi.clear()
							# Keep the currently selected edge
							select_new_edge = False
							continue

				# Callback

				result['status'] = "opti"
				result['indicators'] = deepcopy(indi['crit'])
				result['path'] = origpath
				result['geo_path'] = geo_path
				result['progress'] = overall_progress
				result['edge_clouds'] = deepcopy(prob_clouds)
				result['active_edges'] = deepcopy(seledges)

				if callback: callback(result)

	result['status'] = "done"
	result['progress'] = 1
	if callback: callback(result)

	return result


# node_pos is a dictionary node id -> (lat, lon)
def compute_geo_knn(node_pos, leaf_size=20):
	return {
		'node_ids': list(node_pos.keys()),
		'knn_tree': sklearn.neighbors.BallTree(list(node_pos.values()), leaf_size=leaf_size, metric='pyfunc', func=commons.geodesic)
	}


# Nearest graph edges to a geo-location q = (lat, lon)
def estimate_kne(g, knn, q, ke=10) :
	assert('knn_tree' in knn)
	assert('node_ids' in knn)

	def nearest_nodes(q, kn=10) :
		# Find nearest nodes using the KNN tree
		(dist, ind) = (boo.reshape(-1) for boo in knn['knn_tree'].query(np.asarray(q).reshape(1, -1), k=kn))
		# Return pairs (graph-node-id, distance-to-q) sorted by distance
		return list(zip([knn['node_ids'][i] for i in ind], dist))

	# Locate k nearest directed edges among a set of candidates
	# Return a list of pairs (edge, distance-to-q)
	# Assumes node (lat, lon) coordinates in the node attribute 'pos'
	def filter_nearest_edges(g, q, edges, k):
		return list(heapq.nsmallest(
			k,
			[
				# Attach distance from q
				(e, dist_to_segment(q, (g.nodes[e[0]]['pos'], g.nodes[e[1]]['pos']))[0])
				for e in edges
			],
			key=(lambda ed : ed[1])
		))

	for kn in range(ke, 20*ke, 5) :
		# Get some closest nodes first, then their incident edges
		ee = set(g.edges(nbunch=list(dict(nearest_nodes(q, kn=kn)).keys())))
		# Append reverse edges, just in case
		ee = set(list(ee) + [(b, a) for (a, b) in ee if g.has_edge(b, a)])
		# Continue looking for more edges?
		if (len(ee) < (4*ke)) : continue
		# Filter down to the nearest ones
		ee = filter_nearest_edges(g, q, ee, ke)
		return ee

	raise RuntimeError("Could not find enough nearest edges.")


def foo() :

	mapbox_token = open("../.credentials/UV/mapbox-token.txt", 'r').read()

	osm_graph_file = "../OUTPUT/02/UV/kaohsiung.pkl"

	print("Loading OSM...")
	OSM = pickle.load(open(osm_graph_file, 'rb'))

	print("Retrieving the KNN tree...")

	# Road network (main graph component) with nearest-neighbor tree for the nodes
	g : nx.DiGraph
	(g, knn) = commons.inspect(('g', 'knn'))(OSM['main_component_with_knn'])

	kne = (lambda q : estimate_kne(g, knn, q, ke=20))

	# Free up memory
	del OSM

	#

	# # Get some waypoints
	# routes_file = "../OUTPUT/00/ORIGINAL_MOTC/Kaohsiung/CityBusApi_StopOfRoute.json"
	#
	# motc_routes = commons.index_dicts_by_key(
	# 	commons.zipjson_load(routes_file),
	# 	lambda r: r['RouteUID'],
	# 	preserve_singletons=['Direction', 'Stops']
	# )

	# Waypoints
	# # (route_id, direction) = ('KHH1221', 0)
	# # (route_id, direction) = ('KHH29', 0)
	# # (route_id, direction) = ('KHH38', 0)
	# # (route_id, direction) = ('KHH87', 1)
	# # (route_id, direction) = ('KHH121', 1)
	# (route_id, direction) = ('KHH11', 1)
	# #
	# waypoints = list(map(commons.inspect({'StopPosition': ('PositionLat', 'PositionLon')}), motc_routes[route_id]['Stops'][direction]))

	#waypoints = [(22.622249, 120.368713), (22.621929, 120.367332), (22.622669, 120.367736), (22.623569, 120.366722), (22.624959, 120.364402), (22.625329, 120.36338), (22.625379, 120.362777), (22.62565, 120.361061), (22.62594, 120.359947), (22.62602, 120.354911), (22.62577, 120.351226), (22.625219, 120.34732), (22.62494, 120.3442), (22.624849, 120.34317), (22.62597, 120.342582), (22.626169, 120.344428), (22.62811, 120.344451), (22.62968, 120.33908), (22.63017, 120.337562), (22.63042, 120.336341), (22.631919, 120.331932), (22.632989, 120.327766), (22.632789, 120.325233), (22.632829, 120.324371), (22.633199, 120.32283), (22.633449, 120.321639), (22.63459, 120.31707), (22.636629, 120.314437), (22.63758, 120.308952), (22.6375, 120.307777), (22.637899, 120.301162), (22.63788, 120.298866), (22.637899, 120.297393), (22.63718, 120.294151), (22.636989, 120.293609), (22.6354, 120.288566), (22.635179, 120.287719), (22.634139, 120.284576), (22.632179, 120.28379), (22.631229, 120.283309), (22.628789, 120.28199), (22.62507, 120.28054), (22.624259, 120.282028), (22.622869, 120.284973), (22.62247, 120.285827), (22.623029, 120.286407), (22.62531, 120.28524)]
	#waypoints = [(22.62269, 120.367767), (22.623899, 120.366409), (22.626039, 120.359397), (22.62615, 120.357887), (22.62602, 120.35337), (22.625059, 120.345809), (22.625989, 120.342529), (22.625999, 120.343856), (22.626169, 120.344413), (22.628049, 120.344436), (22.628969, 120.340843), (22.62993, 120.338348), (22.63025, 120.337356), (22.631309, 120.334068), (22.63269, 120.329841), (22.63307, 120.328491), (22.63297, 120.326713), (22.632949, 120.324851), (22.63385, 120.319831), (22.637609, 120.307678), (22.637609, 120.305633), (22.63762, 120.304847), (22.637859, 120.300231), (22.63796, 120.297439), (22.63787, 120.296707), (22.63739, 120.294357), (22.637079, 120.293472), (22.6359, 120.289939), (22.63537, 120.288353), (22.634149, 120.284728), (22.629299, 120.28228), (22.62652, 120.280738), (22.62354, 120.283637), (22.622549, 120.28572), (22.622999, 120.28627), (22.625379, 120.285156)]
	#waypoints = [(22.62202, 120.368789), (22.62198, 120.368133), (22.62191, 120.367233), (22.62384, 120.366401), (22.624929, 120.364402), (22.625329, 120.363342), (22.62593, 120.363357), (22.62569, 120.360771), (22.6261, 120.357803), (22.62601, 120.355743), (22.62578, 120.351692), (22.625539, 120.349586), (22.62494, 120.344642), (22.62515, 120.3423), (22.62598, 120.343742), (22.627559, 120.344482), (22.629569, 120.339309), (22.630359, 120.336929), (22.63124, 120.333846), (22.6322, 120.330856), (22.632869, 120.326393), (22.632879, 120.324172), (22.63344, 120.321502), (22.63418, 120.318351), (22.637369, 120.312362), (22.637639, 120.303802), (22.637779, 120.301971), (22.63787, 120.30104), (22.63775, 120.300231), (22.637859, 120.297416), (22.6373, 120.294448), (22.63697, 120.293418), (22.636289, 120.291076), (22.635129, 120.287742), (22.634969, 120.287078), (22.631259, 120.283332), (22.627559, 120.281402), (22.626689, 120.280967), (22.624849, 120.280937), (22.623979, 120.282623), (22.623739, 120.283187), (22.62317, 120.286453), (22.625259, 120.285423)]
	waypoints = [(22.62203, 120.368293), (22.62195, 120.367401), (22.624559, 120.36515), (22.624929, 120.364448), (22.62585, 120.363113), (22.625549, 120.36177), (22.6261, 120.357627), (22.625509, 120.349677), (22.62503, 120.345596), (22.62589, 120.342307), (22.627979, 120.344459), (22.628539, 120.34201), (22.629989, 120.33805), (22.63025, 120.337219), (22.63211, 120.331581), (22.633039, 120.328659), (22.63307, 120.327308), (22.63294, 120.326156), (22.632989, 120.323699), (22.63342, 120.321418), (22.63743, 120.310119), (22.63755, 120.305641), (22.637639, 120.304267), (22.636949, 120.293319), (22.6355, 120.289062), (22.63454, 120.285987), (22.63076, 120.283088), (22.62968, 120.282478), (22.627229, 120.281188), (22.62647, 120.280693), (22.62516, 120.280387), (22.624099, 120.282401), (22.622669, 120.285308), (22.62313, 120.286369), (22.625169, 120.285667)]

	# commons.seed()
	# mapmatch(waypoints, g, kne, None, stubborn=0.2)
	# commons.seed()
	# mapmatch(waypoints, g, kne, None, stubborn=0.2)
	# return

	#

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

		if (result['status'] == "zero") :
			print("(Preparing)")
			return

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
		img = ax.imshow(mapi, extent=(left, right, bottom, top), interpolation='none', zorder=-100)

		for (n, (y, x)) in enumerate(waypoints) :
			ax.plot(x, y, 'o', c='m', markersize=4)

		(y, x) = zip(*result['geo_path'])
		ax.plot(x, y, 'b--', linewidth=2, zorder=-50)

		for (nc, pc) in enumerate(result['edge_clouds']) :
			m = max(pc.values())
			for (e, p) in pc.items() :
				(y, x) = zip(*[g.nodes[i]['pos'] for i in e])
				c = ('g' if (result['active_edges'][nc] == e) else 'r')
				ax.plot(x, y, '-', c=c, linewidth=2, alpha=p/m, zorder=150)

		ax.axis([left, right, bottom, top])

		plt.pause(0.1)

		# Next figure update
		result['nfu'] = dt.datetime.now() + dt.timedelta(seconds=2)


	print("Connecting clouds...")

	commons.seed()
	result = mapmatch(waypoints, g, kne, mm_callback, stubborn=0.2)

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
