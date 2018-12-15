
# RA, 2018-11-11

from helpers import commons
from helpers import maps

import datetime as dt
import networkx as nx
import numpy as np
import math
import angles
import heapq
import pickle
import random
import sklearn.neighbors
import geopy.distance
import gpxpy.gpx
from copy import deepcopy
from itertools import chain, groupby, product
from sklearn.neighbors import NearestNeighbors

PARAM = {
	'speed' : {
		# In km/h
		'highway' : {
			"motorway" : 50,
			"trunk" : 40,
			"primary" : 30,
			"secondary" : 20,
			"tertiary" : 18,
			"unclassified" : 16,
			"residential" : 12,
			"service" : 8,
			"motorway_link" : 30,
			"trunk_link" : 20,
			"primary_link" : 15,
			"secondary_link" : 10,
			"tertiary_link" : 9
		}
	},

	# Edge attribute for shortest path computation
	'weight' : 'time',
}

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


def dist_to_segment(x, ab, rel_acc=0.05, abs_acc=1) :
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

	while (abs_acc < min(da, db) < geodist(a, b) / rel_acc) :

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


def simple_gpx(waypoints, segments) :
	try :

		gpx = gpxpy.gpx.GPX()

		for (lat, lon) in waypoints :
			gpx.waypoints.append(gpxpy.gpx.GPXWaypoint(latitude=lat, longitude=lon))

		gpx_track = gpxpy.gpx.GPXTrack()
		gpx.tracks.append(gpx_track)

		for segment in segments :
			gpx_segment = gpxpy.gpx.GPXTrackSegment()

			for (p, q) in segment :
				gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=p, longitude=q))

			gpx_track.segments.append(gpx_segment)

		return gpx

	except Exception as e :
		print("Warning: making GPX object failed ({})".format(e))
		return None


def mapmatch(
		waypoints: list,
		g: nx.DiGraph,
		kne: callable,
		callback: callable = None,
		stubborn: float = 1
):
	""" Find a plausible bus route that visits the waypoints.

	(TODO)
	Returns:
		A dictionary with the following keys
			- 'path' is a list of nodes of the graph g, as an estimate of the route
			- 'geo_path' is a list of (lat, lon) coordinates of those nodes
			- 'edge_clouds' is a list of edge clouds, one for each waypoint
			- 'active_edges' is a list of currently selected edges, one for each edge cloud
	"""

	# Check connectivity of the graph
	if (g.number_of_nodes() > max(map(len, nx.strongly_connected_components(g)))) :
		raise RuntimeError("The graph appears not to be strongly connected.")

	# Dictionary to pass to the callback function (updated below)
	result = { 'waypoints' : waypoints, 'mapmatcher_version' : 11111128 }

	# Before doing anything
	result['status'] = "zero"
	if callback : callback(result)

	# A turn of 90 degrees takes on average 30 seconds (wild guess)
	# Note: could distinguish left and right turns here
	def time_from_angle(a) :
		return (30 * abs(a) / 90) if (abs(a) < 150) else 120
	# Convert the busstop-road distance to "equivalent" bus travel time
	crittime_from_meter_stop = (lambda d : (d / 1.5) * 5)

	# Optimization criterion (lower is better)
	def opticrit(indi):
		return stubborn * crittime_from_meter_stop(indi['miss']) + indi['time']

	# Threshold turn for introducing "Detailed decision node clusters"
	ddnc_threshold = 30

	# Acceptance threshold (between 0 and 1) for edge confidence in an edge cloud
	final_acceptance_threshold = 0.98

	# The graph will be modified
	g = deepcopy(g)

	# Check for the existence of those attributes
	assert(nx.get_node_attributes(g, 'pos')) # Node (lat, lon)
	assert(nx.get_edge_attributes(g, 'len')) # Edge lengths (meters)

	for (a, b, d) in g.edges.data() :
		g[a][b][PARAM['weight']] = d['len'] / (PARAM['speed']['highway'][d['highway']] / 3.6)

	# Mark the original nodes as basenodes
	for n in g.nodes : g.nodes[n]['basenode'] = n
	# Function to express edges in terms of the basenodes
	to_basenodes = (lambda edges : [tuple(g.nodes[n]['basenode'] for n in e) for e in edges])

	# A cloud of candidate edges for each waypoint
	dist_clouds = [dict(kne(wp)) for wp in waypoints]

	# Check if there are edge repeats within clouds
	for dc in dist_clouds :
		if not commons.all_distinct(dc.keys()) :
			print("Warning: Repeated edges in cloud:", sorted(dc.keys()))

	# Initial likelihood prop to road class and inv-prop to the regularized distance (in meters) to the waypoint
	def dist2prob(dc) :
		return { e : (PARAM['speed']['highway'][g[e[0]][e[1]]['highway']] / (1 + d)) for (e, d) in dc.items() }

	# Edge clouds containing relative suitability of edges for the optimal solution
	prob_clouds = [ dist2prob(dc) for dc in dist_clouds ]

	# Intermediate edges -- random initial condition
	seledges = [commons.random_subset(list(pc.keys()), weights=list(pc.values()), k=1).pop() for pc in prob_clouds]

	# Route quality indicators
	indi = dict()

	# Partial shortest path cache, keyed by tuples (start-node, end-node)
	sps_way = dict()

	# Replace node by a "Detailed decision node cluster"
	def make_ddnc(nb) :

		# Incoming and outgoing edges
		(iedges, oedges) = (list(f(nbunch=nb)) for f in [g.in_edges, g.out_edges])

		# Make the cluster
		for (ie, oe) in product(iedges, oedges):
			# New hyperedge between (new) hypernodes
			g.add_edge(ie, oe, len=0, time=time_from_angle(angle(*(g.nodes[n]['pos'] for n in [ie[0], nb, oe[1]]))))

		# Interface edges (old, new)
		e2E = dict([(ie, (ie[0], ie)) for ie in iedges] + [(oe, (oe, oe[1])) for oe in oedges])

		for (e, E) in e2E.items():
			# Refer new nodes to the basenode, and copy the geolocation
			for attr in ['basenode', 'pos'] :
				g.nodes[e][attr] = g.nodes[nb][attr]
			# Replace the interface edges, keeping the edge data
			g.add_edge(*E, **g.get_edge_data(*e))

		# Remove old node and incident edges
		g.remove_node(nb)

		# Now fix the caches

		# Remove invalidated shortest paths
		for (ab, way) in list(sps_way.items()) :
			if nb in way :
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


	def get_current_path(seledges) :

		# Precompute shortest path for each gap
		try:
			for ((_, a), (b, _)) in zip(seledges, seledges[1:]):
				if (a, b) not in sps_way:
					sps_way[(a, b)] = nx.shortest_path(g, source=a, target=b, weight=PARAM['weight'])
		except nx.NetworkXNoPath:
			raise

		# Non-repeating edges
		nre = commons.remove_repeats(seledges)

		# Construct a path through the selected edges
		path = [nre[0][0]] + list(chain.from_iterable(sps_way[(a, b)] for ((_, a), (b, _)) in zip(nre, nre[1:]))) + [nre[-1][-1]]

		# Convert node IDs to (lat, lon) coordinates
		# Note: must allow possible repeats (deal later)
		geo_path = [g.nodes[n]['pos'] for n in path]

		# Undo "Detailed decision node cluster"s for the user
		# Note: must allow possible repeats (deal later)
		origpath = [g.nodes[n]['basenode'] for n in path]

		return (geo_path, origpath, path)


	# Intialized
	result['status'] = "init"
	if callback : callback(result)

	# Collect the best solution variants here
	elite = []

	# Optimization rounds: first global then local
	resets = [[]]
	resets += [range(n, n + 2) for n in range(len(prob_clouds) - 1)]
	resets += [range(n, n + 2) for n in range(len(prob_clouds) - 1)]

	for (progress_reset, reset) in enumerate(resets) :
		progress_reset = progress_reset / len(resets)

		for nc in reset :
			prob_clouds[nc] = dist2prob(dist_clouds[nc])

		while True :
			# Note: the variable nc is reserved throughout this loop for "the number of the currently selected edge cloud"

			# Choose a random edge cloud, preferably the "least solved" one
			nc = commons.random_subset(range(len(prob_clouds)), weights=[(sum(pc.values()) - pc[e]) for (e, pc) in zip(seledges, prob_clouds)], k=1).pop()

			# Cloud edges with weights
			(ce, cw) = zip(*[(e, p) for (e, p) in prob_clouds[nc].items() if (e != seledges[nc])])

			# Choose a candidate edge from the cloud
			seledges[nc] = commons.random_subset(ce, weights=cw, k=1).pop()

			# Reconstruct the route from intermediate edges
			(geo_path, origpath, path) = get_current_path(seledges)

			# Introduce turn penalization
			if True :

				# Candidates for "detail decision node clusters"
				cand_ddnc = set(
					b
					for (p, q, r, b) in zip(geo_path, geo_path[1:], geo_path[2:], path[1:])
					# Check if the node is original and the turn is significant
					if (g.nodes[b]['basenode'] == b) and (abs(angle(p, q, r)) >= ddnc_threshold)
				)

				if cand_ddnc :
					for nb in cand_ddnc :
						make_ddnc(nb)

					# Changes in the graph invalidate previous solutions/metrics
					elite.clear()
					indi.clear()

					continue

			# Previous value of the quality indicators
			old_indi = indi.copy()

			# Indicator 1: Sum of distances of the selected edges to waypoints
			indi['miss'] = sum(dc[e] for (e, dc) in zip(seledges, dist_clouds))

			# Indicator 2: The total travel time for the path
			indi['time'] = sum(g.get_edge_data(*e)[PARAM['weight']] for e in zip(path, path[1:]))

			# Optimization criterion (lower is better)
			indi['crit'] = opticrit(indi)

			# Re-weight the current edge in its cloud
			if not old_indi : continue
			rel_delta = indi['crit'] / old_indi['crit']
			prob_clouds[nc][seledges[nc]] *= (1.8 if (rel_delta < 1) else (0.7 / rel_delta))

			# Update the "elite" solution variants
			elite = sorted(set(elite + [(indi['crit'], tuple(seledges))]), key=(lambda se : se[0]))[0:5]

			# Normalize to avoid flop errors
			prob_clouds = [{ e : (p / sum(pc.values())) for (e, p) in pc.items() } for pc in prob_clouds]

			# Status
			if callback:
				result['status'] = "opti"
				result['progress'] = progress_reset * np.mean([min(1, max(pc.values()) / sum(pc.values()) / final_acceptance_threshold) for pc in prob_clouds])
				(result['geo_path'], result['path'], _) = map(list, zip(*commons.remove_repeats(zip(*get_current_path(seledges)))))

				result['(indicators)'] = deepcopy(indi['crit'])
				result['(edge_clouds)'] = [ dict(zip(to_basenodes(pc.keys()), pc.values())) for pc in prob_clouds ]
				result['(active_edges)'] = to_basenodes(seledges)

				callback(result)

			# Optimization loop termination attempt
			if all((pc[e] / sum(pc.values()) >= final_acceptance_threshold) for (e, pc) in zip(seledges, prob_clouds)) :

				# Revert to the best solution seen so far
				seledges = list(elite[0][1])

				# Remove transient indicators
				result = { k : v for (k, v) in result.items() if not ((k[0] == "(") and (k[-1] == ")")) }

				break

	# Reconstruct the route from the selected edges
	result['status'] = "done"
	result['progress'] = 1
	(result['geo_path'], result['path'], _) = map(list, zip(*commons.remove_repeats(zip(*get_current_path(seledges)))))

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
	#waypoints = [(22.62203, 120.368293), (22.62195, 120.367401), (22.624559, 120.36515), (22.624929, 120.364448), (22.62585, 120.363113), (22.625549, 120.36177), (22.6261, 120.357627), (22.625509, 120.349677), (22.62503, 120.345596), (22.62589, 120.342307), (22.627979, 120.344459), (22.628539, 120.34201), (22.629989, 120.33805), (22.63025, 120.337219), (22.63211, 120.331581), (22.633039, 120.328659), (22.63307, 120.327308), (22.63294, 120.326156), (22.632989, 120.323699), (22.63342, 120.321418), (22.63743, 120.310119), (22.63755, 120.305641), (22.637639, 120.304267), (22.636949, 120.293319), (22.6355, 120.289062), (22.63454, 120.285987), (22.63076, 120.283088), (22.62968, 120.282478), (22.627229, 120.281188), (22.62647, 120.280693), (22.62516, 120.280387), (22.624099, 120.282401), (22.622669, 120.285308), (22.62313, 120.286369), (22.625169, 120.285667)]
	#waypoints = [(22.666889, 120.358613), (22.666389, 120.358893), (22.667886, 120.357973), (22.669096, 120.35728), (22.672, 120.356413), (22.673586, 120.356866), (22.67395, 120.35764), (22.670996, 120.359653), (22.669636, 120.360426), (22.667536, 120.361346), (22.665766, 120.361893), (22.663703, 120.362173), (22.661463, 120.362533), (22.66128, 120.363266), (22.659683, 120.364013), (22.65876, 120.361999), (22.658496, 120.360746), (22.656686, 120.360226), (22.653473, 120.359653), (22.650909, 120.359719), (22.650743, 120.357439), (22.65097, 120.356733), (22.650973, 120.355746), (22.651303, 120.351026), (22.651933, 120.349693), (22.65304, 120.349733), (22.652703, 120.349173), (22.651806, 120.348826), (22.65078, 120.348439), (22.649753, 120.348079), (22.643733, 120.346253), (22.642279, 120.345799), (22.642226, 120.343906), (22.642313, 120.343186), (22.642483, 120.342386), (22.639283, 120.340866), (22.639399, 120.340186), (22.639863, 120.338213), (22.640733, 120.332533), (22.639569, 120.3322), (22.638956, 120.332173), (22.639066, 120.328613), (22.639433, 120.326866), (22.639306, 120.326026), (22.6397, 120.322373), (22.639946, 120.319919), (22.640243, 120.316946), (22.637166, 120.311866), (22.637503, 120.306773), (22.63753, 120.304773), (22.637616, 120.304079), (22.637709, 120.302853), (22.636179, 120.302306), (22.635326, 120.302373), (22.634396, 120.302373), (22.631286, 120.301599), (22.630943, 120.300679), (22.630089, 120.298066), (22.628786, 120.294079), (22.627633, 120.290586), (22.62723, 120.289253), (22.62659, 120.287533), (22.62601, 120.286213), (22.626173, 120.28564), (22.624866, 120.285866), (22.623696, 120.2866), (22.623153, 120.286386), (22.621536, 120.284959), (22.620976, 120.284999)]
	#waypoints = [(22.666556, 120.358786), (22.66751, 120.35836), (22.668576, 120.357613), (22.672313, 120.356319), (22.673593, 120.356906), (22.665803, 120.361893), (22.663703, 120.362239), (22.662706, 120.362399), (22.659723, 120.363933), (22.658963, 120.363186), (22.658523, 120.360759), (22.6508, 120.357293), (22.65096, 120.356733), (22.650733, 120.355213), (22.650686, 120.352519), (22.652966, 120.34976), (22.645576, 120.346626), (22.643599, 120.346106), (22.641999, 120.345106), (22.642226, 120.343879), (22.642373, 120.343119), (22.642366, 120.341933), (22.640166, 120.341533), (22.639253, 120.341279), (22.639373, 120.34028), (22.639679, 120.338506), (22.640713, 120.332719), (22.63971, 120.33216), (22.638663, 120.332093), (22.638593, 120.331199), (22.638933, 120.32952), (22.639263, 120.326546), (22.63934, 120.325826), (22.639453, 120.324773), (22.639633, 120.323279), (22.63988, 120.320959), (22.640173, 120.317826), (22.637606, 120.3048), (22.637653, 120.304106), (22.637756, 120.302306), (22.636829, 120.302213), (22.636233, 120.302199)]
	#waypoints = [(22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60642, 120.338256), (22.60651, 120.338188), (22.606119, 120.338569), (22.606109, 120.33834), (22.605649, 120.333862), (22.60655, 120.33345), (22.60831, 120.333213), (22.617879, 120.332069), (22.619409, 120.331848), (22.61968, 120.33184), (22.622409, 120.331458), (22.622329, 120.329803), (22.622289, 120.329437), (22.62141, 120.327056), (22.62124, 120.326507), (22.62095, 120.325752), (22.62008, 120.323219), (22.618999, 120.319877), (22.61882, 120.319168), (22.61853, 120.318038), (22.617969, 120.316169), (22.617969, 120.316169), (22.617319, 120.314498), (22.61683, 120.313072), (22.616478999999998, 120.31208), (22.616478999999998, 120.31208), (22.615419, 120.308982), (22.615159, 120.30812), (22.61498, 120.307182), (22.61498, 120.307182), (22.614429, 120.305999), (22.614429, 120.305999), (22.614099, 120.305229), (22.61411, 120.30516), (22.61375, 120.303573), (22.61359, 120.303199), (22.61359, 120.303199), (22.613389, 120.302856), (22.613109, 120.301681), (22.612779, 120.301078), (22.61264, 120.300666), (22.61255, 120.300239), (22.6123, 120.298408), (22.61313, 120.29811), (22.621259, 120.295913), (22.62136, 120.296142), (22.62174, 120.296768), (22.622539, 120.29956), (22.622659, 120.299919), (22.62294, 120.300529), (22.62302, 120.300811), (22.62351, 120.301116), (22.624648, 120.301299), (22.6268, 120.301521)]

	# waypoints = [(22.60642, 120.338256), (22.605649, 120.333862), (22.60655, 120.33345), (22.60831, 120.333213), (22.617879, 120.332069), (22.619409, 120.331848), (22.622409, 120.331458), (22.622329, 120.329803), (22.62141, 120.327056), (22.62095, 120.325752), (22.62008, 120.323219), (22.618999, 120.319877), (22.61882, 120.319168), (22.61853, 120.318038), (22.617969, 120.316169), (22.617319, 120.314498), (22.61683, 120.313072), (22.616478999999998, 120.31208), (22.615419, 120.308982), (22.615159, 120.30812), (22.61498, 120.307182), (22.614429, 120.305999), (22.614099, 120.305229), (22.61375, 120.303573), (22.613389, 120.302856), (22.613109, 120.301681), (22.612779, 120.301078), (22.61255, 120.300239), (22.6123, 120.298408), (22.61313, 120.29811), (22.621259, 120.295913), (22.62174, 120.296768), (22.622539, 120.29956), (22.62294, 120.300529), (22.62351, 120.301116), (22.624648, 120.301299), (22.6268, 120.301521)]
	# wp1 = (22.61313, 120.29811) # lower
	# wp2 = (22.621259, 120.295913) # upper
	# print(waypoints.index(wp1), waypoints.index(wp2)) # 29 30
	
	waypoints = [(22.591286, 120.305706), (22.593253, 120.305906), (22.59474, 120.305306), (22.597746, 120.304186), (22.596886, 120.305133)]

	# waypoints = [(22.613109, 120.301681), (22.612779, 120.301078), (22.61255, 120.300239), (22.6123, 120.298408), (22.61313, 120.29811), (22.621259, 120.295913), (22.62174, 120.296768), (22.622539, 120.29956)]

	# commons.seed()
	# mapmatch(waypoints, g, kne, None, stubborn=0.2)
	# commons.seed()
	# mapmatch(waypoints, g, kne, None, stubborn=0.2)
	# return

	#

	# TODO: abort if nearest edges are too far
	# TODO: no path to node issue

	# Includes renderer selection:
	import matplotlib.pyplot as plt

	def mm_callback(result) :

		if (result['status'] == "zero") :
			print("(Preparing)")

		if (result['status'] == "init") :
			print("(Optimizing)")

		if (result['status'] == "opti") :
			if (dt.datetime.now() < result.get('nfu', dt.datetime.min)) :
				return

		if (result['status'] == "done") :
			print("(Done)")

		if (result['status'] == "zero") :
			fig : plt.Figure
			ax : plt.Axes
			(fig, ax) = plt.subplots()

			for (n, (y, x)) in enumerate(result['waypoints']):
				ax.plot(x, y, 'bo')

			ax.axis(commons.niceaxis(ax.axis(), expand=1.1))
			ax.autoscale(enable=False)

			# Download the background map
			mapi = maps.get_map_by_bbox(maps.ax2mb(*ax.axis()), token=mapbox_token)

			result['plt'] = { 'fig' : fig, 'ax' : ax, 'map' : mapi, 'bbox' : ax.axis() }

		if (result['status'] in ["zero", "opti", "done"]) :

			fig : plt.Figure
			ax : plt.Axes
			(fig, ax, mapi, bbox) = commons.inspect({'plt': ('fig', 'ax', 'map', 'bbox')})(result)

			# Clear the axes
			ax.cla()

			# Apply the background map
			img = ax.imshow(mapi, extent=bbox, interpolation='none', zorder=-100)

			for (n, (y, x)) in enumerate(result['waypoints']) :
				ax.plot(x, y, 'o', c='m', markersize=4)

			if ('geo_path' in result) :
				(y, x) = zip(*result['geo_path'])
				ax.plot(x, y, 'b--', linewidth=2, zorder=-50)

			if ('(edge_clouds)' in result) :
				for (nc, pc) in enumerate(result['(edge_clouds)']) :
					for (e, p) in pc.items() :
						(y, x) = zip(*[g.nodes[i]['pos'] for i in e])
						c = ('g' if (result['(active_edges)'][nc] == e) else 'r')
						ax.plot(x, y, '-', c=c, linewidth=2, alpha=(p / max(pc.values())), zorder=150)

			ax.axis(bbox)

			plt.pause(0.1)

		if (result['status'] == "opti") :
			# Next figure update
			result['nfu'] = dt.datetime.now() + dt.timedelta(seconds=4)

		if (result['status'] == "done") :
			open("result.gpx", 'w').write(simple_gpx(result['waypoints'], [result['geo_path']]).to_xml())

	print("Calling mapmatch...")

	plt.ion()

	commons.seed()
	result = mapmatch(waypoints, g, kne, mm_callback, stubborn=0.2)

	for _ in range(20) :
		result = mapmatch(waypoints, g, kne, mm_callback, stubborn=0.2)
		print(result['path'])
		plt.pause(5)

	plt.ioff()
	plt.show()

	return


if (__name__ == "__main__") :
	foo()
