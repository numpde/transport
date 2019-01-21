
# RA, 2018-11-11

from helpers import commons
from helpers import maps

import datetime as dt
import networkx as nx
import numpy as np
import math
import time
import angles
import heapq
import pickle
import random
import sklearn.neighbors
import geopy.distance
import gpxpy.gpx
import traceback

from copy import deepcopy
from itertools import chain, groupby, product
from sklearn.neighbors import NearestNeighbors

from progressbar import progressbar

from joblib import delayed, Parallel

from typing import Generator, Union, Tuple


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

	# Expect to find at least one road within this radius from each waypoint
	'max_wp_to_graph_dist' : 30, # meters

	# Prefer paths with fewer turns?
	'use_turn_penalization' : True,

	# Discard waypoints that are too close
	'waypoints_min_distance' : 60,  # (meters)

	# Minimum number of waypoints to consider
	'waypoints_min_number' : 4,

	# When mapmatching by subgroups...
	'waypoints_groupsize' : 9,

	# Mapmatch subgroups in random order?
	'waypoints_mapmatch_shuffle' : True,

	# Maximum number of subgroups to mapmatch (should enable shuffle)
	'waypoints_mapmatch_max_groups' : 512,

	# Extract a subgraph around the waypoints for mapmatching
	'do_graph_extract' : True,

	'#jobs' : commons.cpu_frac(0.7),
}


# Generic error class for this module
class MapMatchingError(Exception) :
	pass


# Signed "turning" angle (p, q) to (q, r) in degrees,
# where p, q, r are (lat, lon) coordinates
# https://stackoverflow.com/a/16180796/3609568
def angle(p, q, r) -> float :
	# Note: the +plus+ is a concatenation of tuples here
	pq = angles.bear(*map(angles.d2r, p + q))
	qr = angles.bear(*map(angles.d2r, q + r))
	return (((angles.r2d(pq - qr) + 540) % 360) - 180)


# Approximate distance to segment using flat geometry
# Returns the distance d >= 0 and the relative coordinate 0 <= t <= 1 of the nearest point
def dist_to_segment(x, ab, distance=geopy.distance.great_circle) -> Tuple[float, float] :
	# Compute lengths
	xa = distance(x, ab[0]).m
	xb = distance(x, ab[1]).m
	ab = distance(ab[0], ab[1]).m
	if (ab == 0) : return (xa, 0)
	# Heron's formula for the area-squared
	s = (xa + xb + ab) / 2
	AA = s * (s - xa) * (s - xb) * (s - ab)
	# Height
	h = math.sqrt(max(0, AA)) / ab * 2
	# From base edngpoints to base of the height
	ah = math.sqrt(max(0, xa**2 - h**2))
	bh = math.sqrt(max(0, xb**2 - h**2))
	# Distance and relative coordinate
	(d, t) = max([(bh, (xa, 0)), (ah, (xb, 1)), (ab, (h, ah / ab))])[1]
	return (d, t)


# Make a GPX object with waypoints and one track
def simple_gpx(waypoints, segments) -> Union[gpxpy.gpx.GPX, None] :
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


# Keep a certain distance between waypoints (in meters)
def sparsified(wps, dist=0.0) -> Generator :
	a = next(iter(wps))
	yield a
	for b in wps :
		if (commons.geodesic(a, b) >= dist) :
			a = b
			yield a


class NodeGenerator :
	def __init__(self, g: nx.DiGraph) :
		self.g = g
		self.n = 1
	def __iter__(self) :
		return self
	def __next__(self) :
		while self.g.has_node(self.n) :
			self.n += 1
		self.g.add_node(self.n)
		return self.n


# Return the largest connected component as a new graph
# Uses the copy() method of the graph but does not deep-copy attributes
def largest_component(g: nx.DiGraph, components=nx.strongly_connected_components) -> nx.DiGraph :
	return nx.subgraph(g, max(list(components(g)), key=len)).copy()

# Split long edges
# Assumes edge length in the edge attribute 'len'
# Assumes node geo-location in the node attribute 'pos'
# Assumes len(u, v) == len(v, u)
def break_long_edges(g: nx.DiGraph, max_len=50, node_generator=None) -> None :

	from math import ceil

	lens = dict(nx.get_edge_attributes(g, 'len'))
	assert(lens), "Expect edge lengths in the 'len' attribute"

	# Edges to split
	edges = { e for (e, s) in lens.items() if (s > max_len) }
	commons.logger.debug("Number of edges to split is {}".format(len(edges)))

	# Edges are identified and grouped if they connect the same nodes
	iso = (lambda e : (min(e), max(e)))
	edges = [ list(g) for (__, g) in commons.sort_and_group(edges, iso) ]
	commons.logger.debug("Number of edge groups to split is {}".format(len(edges)))

	# Get default node ID generator?
	node_generator = node_generator or NodeGenerator(g)

	def split(e, bothways: bool) :
		# Edge to partition
		(a, b) = e
		assert(g.has_edge(a, b)), "Edge not in graph"

		len_key = 'len'
		pos_key = 'pos'

		# Number of new nodes
		nnn = ceil(g.edges[a, b][len_key] / max_len) - 1

		# Need to partition?
		if not nnn : return

		# Relative coordinates of all nodes
		tt = np.linspace(0, 1, 1 + nnn + 1)

		# All new edges have the same length (also the reverse ones)
		new_len = g.edges[a, b][len_key] / (len(tt) - 1)

		if bothways :
			assert(abs(g.edges[b, a][len_key] - g.edges[a, b][len_key]) <= 1), "Gross back and forth edge length mismatch"

		# All nodes along the old edge, old--new--old
		all_nodes = [a] + list(next(node_generator) for __ in range(nnn)) + [b]

		# All new edges
		new_edges = set(zip(all_nodes, all_nodes[1:]))
		if bothways : new_edges |= set(zip(all_nodes[1:], all_nodes))

		# Add the nodes and edges to the graph, copy attributes, overwrite 'len'
		g.add_edges_from(new_edges, **{**g.edges[a, b], 'len': new_len})

		# Geo-location of new nodes
		all_pos = map(tuple, zip(*(np.outer(g.nodes[a][pos_key], 1 - tt) + np.outer(g.nodes[b][pos_key], tt))))

		# Set the locations of all nodes
		nx.set_node_attributes(g, dict(zip(all_nodes, all_pos)), pos_key)

		# Remove old edge(s)
		try :
			g.remove_edge(a, b)
			g.remove_edge(b, a)
		except nx.NetworkXError :
			pass

		# if 11172 in all_nodes :
		# 	commons.logger.debug("e = {}; all_nodes = {}; new_edges = {}".format(e, all_nodes, new_edges))

	for ee in edges :
		split(next(iter(ee)), bothways=(len(ee) == 2))


# MAIN COMPUTATION, BASED ON THE INPUT IN 'presult'
def mapmatch_complete_this(presult) :

	many_partial = presult['many_partial']
	stubborn = presult['stubborn']

	g1 = presult['(g)']
	dist_clouds = presult['(dist_clouds)']
	callback = presult['(callback)']

	presult['status'] = "zero"
	presult['progress'] = 0

	try :
		del presult['path']
		del presult['geo_path']
	except :
		pass

	# A turn of 90 degrees takes on average 30 seconds (wild guess)
	# Note: could distinguish left and right turns here
	def time_from_angle(a) :
		return (30 * abs(a) / 90) if (abs(a) < 150) else 120

	# Convert the busstop-road distance to "equivalent" bus travel time
	def crittime_from_meter_stop(d) :
		return (d / 1.5) * 5

	# Optimization criterion (lower is better)
	def opticrit(indi):
		return stubborn * crittime_from_meter_stop(indi['miss']) + indi['time']

	# Threshold turn for introducing "Detailed decision node clusters"
	ddnc_threshold = 30

	# Acceptance threshold (between 0 and 1) for edge confidence in an edge cloud
	final_acceptance_threshold = 0.98

	# Function to express edges in terms of the basenodes
	to_basenodes = (lambda edges : [tuple(g1.nodes[n]['basenode'] for n in e) for e in edges])

	# Initial likelihood prop to road class and inv-prop to the regularized distance (in meters) to the waypoint
	def dist2prob(dc) :
		return {e : (PARAM['speed']['highway'][g1.edges[e[0], e[1]]['highway']] / (1 + d)) for (e, d) in dc.items()}

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
		(iedges, oedges) = (list(f(nbunch=nb)) for f in [g1.in_edges, g1.out_edges])

		# Make the cluster
		for (ie, oe) in product(iedges, oedges):
			# New hyperedge between (new) hypernodes
			g1.add_edge(ie, oe, **{'len' : 0, PARAM['weight'] : time_from_angle(angle(*(g1.nodes[n]['pos'] for n in [ie[0], nb, oe[1]])))})

		# Interface edges (old, new)
		e2E = { **{ ie: (ie[0], ie) for ie in iedges }, **{ oe : (oe, oe[1]) for oe in oedges } }

		for (e, E) in e2E.items():
			# Copy all attributes, i.e. geo-location and the original 'basenode'
			g1.add_node(e, **g1.nodes[nb])
			# Replace the interface edges, keeping the edge data
			g1.add_edge(*E, **g1.get_edge_data(*e))

		# Remove old node and incident edges
		g1.remove_node(nb)

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
			for (pc, dc) in zip(prob_clouds, dist_clouds) :
				try :
					pc[E] = pc.pop(e)
					dc[E] = dc.pop(e)
				except KeyError :
					pass


	def get_current_path(seledges) :

		# Precompute shortest path for each gap
		try:
			for ((_, a), (b, _)) in zip(seledges, seledges[1:]):
				if (a, b) not in sps_way:
					try :
						sps_way[(a, b)] = nx.shortest_path(g1, source=a, target=b, weight=PARAM['weight'])
					except nx.NodeNotFound :
						commons.logger.error("Path {}->{} failed".format(a, b))
						raise
		except nx.NetworkXNoPath:
			raise

		# Non-repeating edges
		nre = commons.remove_repeats(seledges)

		# Construct a path through the selected edges
		path = [nre[0][0]] + list(chain.from_iterable(sps_way[(a, b)] for ((_, a), (b, _)) in zip(nre, nre[1:]))) + [nre[-1][-1]]

		# Convert node IDs to (lat, lon) coordinates
		# Note: must allow possible repeats (deal later)
		geo_path = [g1.nodes[n]['pos'] for n in path]

		# Undo "Detailed decision node cluster"s for the user
		# Note: must allow possible repeats (deal later)
		origpath = [g1.nodes[n]['basenode'] for n in path]

		return (geo_path, origpath, path)


	# Intialized
	presult['status'] = "init"
	if callback : callback(presult)

	# Collect the best solution variants here
	elite = []

	# Optimization rounds: first global ...
	resets = [[]]

	# ... then local
	if not many_partial :
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
			if PARAM['use_turn_penalization'] :

				# Candidates for "detail decision node clusters"
				cand_ddnc = set(
					b
					for (p, q, r, b) in zip(geo_path, geo_path[1:], geo_path[2:], path[1:])
					# Check if the node is original and the turn is significant
					if (g1.nodes[b]['basenode'] == b) and (abs(angle(p, q, r)) >= ddnc_threshold)
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
			indi['time'] = sum(g1.get_edge_data(*e)[PARAM['weight']] for e in zip(path, path[1:]))

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

			# Intermediate status
			if callback :
				presult['status'] = "opti"
				presult['progress'] = progress_reset * np.mean([min(1, max(pc.values()) / sum(pc.values()) / final_acceptance_threshold) for pc in prob_clouds])
				(presult['geo_path'], presult['path'], _) = map(list, zip(*commons.remove_repeats(zip(*get_current_path(seledges)))))

				presult['(indicators)'] = deepcopy(indi['crit'])
				presult['(edge_clouds)'] = [dict(zip(to_basenodes(pc.keys()), pc.values())) for pc in prob_clouds]
				presult['(active_edges)'] = to_basenodes(seledges)

				callback(presult)

			# Optimization loop termination attempt
			if all((pc[e] / sum(pc.values()) >= final_acceptance_threshold) for (e, pc) in zip(seledges, prob_clouds)) :

				# Revert to the best solution seen so far
				seledges = list(elite[0][1])

				# Optimization finished
				break

	# Final status. Reconstruct the route from the selected edges
	presult['status'] = "done"
	presult['progress'] = 1
	(presult['geo_path'], presult['path'], _) = map(list, zip(*commons.remove_repeats(zip(*get_current_path(seledges)))))

	# Remove transient indicators
	presult = {k : v for (k, v) in presult.items() if not ((k[0] == "(") and (k[-1] == ")"))}

	if callback: callback(presult)

	return presult


def mapmatch(
		waypoint_sets: dict,
		g0: nx.DiGraph,
		kne: callable,
		knn = None,
		callback: callable = None,
		stubborn: float = 1,
		many_partial: bool = False
) -> Generator :
	"""
	Find a plausible bus route or pieces of it along the waypoints.

	(TODO)
	Returns:
		A dictionary with the following keys
			- 'path' is a list of nodes of the graph g0, as an estimate of the route
			- 'geo_path' is a list of (lat, lon) coordinates of those nodes
			- 'edge_clouds' is a list of edge clouds, one for each waypoint
			- 'active_edges' is a list of currently selected edges, one for each edge cloud
	"""

	# Dictionary to collect status and result (updated continuously)
	result = { 'waypoint_sets' : waypoint_sets, 'mapmatcher_version' : 11111143 }

	# Before doing anything
	result['status'] = "zero"
	if callback : callback(result)

	# Check connectivity of the graph
	if (g0.number_of_nodes() > max(map(len, nx.strongly_connected_components(g0)))) :
		raise MapMatchingError("The graph appears not to be strongly connected")

	# The graph will be modified from here on
	# g0 = deepcopy(g0)
	# commons.logger.warning("Attributes added to the graph supplied to mapmatch")

	# Check for the existence of those attributes
	assert(nx.get_node_attributes(g0, 'pos')) # Node (lat, lon)
	assert(nx.get_edge_attributes(g0, 'len')) # Edge lengths (meters)

	# Nearest neighbors
	if not knn :
		knn = compute_geo_knn(nx.get_node_attributes(g0, 'pos'))

	# Original weights for shortest path computation
	g0_weights = {
		(a, b) : d['len'] / (PARAM['speed']['highway'][d['highway']] / 3.6)
		for (a, b, d) in g0.edges.data()
	}


	commons.logger.debug("Computing waypoint nearest edges...")

	# Waypoints' nearest edges: wp --> (dist_cloud : edge --> distance)
	def get_wp2ne(wpts) :
		return {wp: dict(kne(wp)) for wp in set(wpts)}

	# Compute the nearest edges map for each waypoint set
	waypoints_kne = {
		setid: get_wp2ne(wp2ne)
		for (setid, wp2ne) in progressbar(waypoint_sets.items())
	}

	# Keep only those that are "inside" the graph
	waypoints_kne = {
		setid: wp2ne
		for (setid, wp2ne) in waypoints_kne.items()
		if (max(min(dd.values()) for dd in wp2ne.values()) <= PARAM['max_wp_to_graph_dist'])
	}

	# Waypoints from all sets
	result['waypoints_all'] = list(chain.from_iterable(wp2ne.keys() for wp2ne in waypoints_kne.values()))

	if not result['waypoints_all'] :
		raise MapMatchingError("No waypoints near the graph")

	# CHECK if there are edge repeats within clouds
	for dc in waypoints_kne.values() :
		if not commons.all_distinct(dc.keys()) :
			commons.logger.warning("Repeated edges in cloud: {}".format(sorted(dc.keys())))

	# Make pairs (setid, waypoint_group)
	# using waypoints_kne and waypoint_sets
	def split_into_groups() :
		for (setid, wp2ne) in waypoints_kne.items() :

			# Waypoints of this set w/o consecutive repeats
			waypoints = commons.remove_repeats(waypoint_sets[setid])

			if many_partial :
				# Avoid getting stuck in places of dense waypoints
				waypoints = list(sparsified(waypoints, dist=(PARAM['waypoints_min_distance'] / 10)))

				# Extract groups of consecutive waypoints
				groups = [
					list(sparsified(waypoints[k:], dist=PARAM['waypoints_min_distance']))[0:PARAM['waypoints_groupsize']]
					for k in range(0, len(waypoints), round(PARAM['waypoints_groupsize'] / 3))
				]

				# Remove too-small groups
				groups = [ g for g in groups if (len(g) >= PARAM['waypoints_min_number']) ]

				# Remove redundant groups
				k = 1
				while (k < len(groups)) :
					if set(groups[k]).issubset(set(groups[k - 1])) :
						groups.pop(k)
					else :
						k += 1

				commons.logger.info("From set {}, extracted {} waypoint subgroups".format(setid, len(groups)))

				# Mapmatch on the subgroups
				# Note: the loop could be empty
				for wpts in groups :
					yield (setid, wpts)

			else :
				# Do not split waypoints into subgroups
				yield (setid, sparsified(waypoints, dist=PARAM['waypoints_min_distance']))


	# List of pairs (setid, waypoint_group)
	commons.logger.debug("Grouping waypoints into subgroups...")
	groups = list(split_into_groups())

	if not groups :
		raise MapMatchingError("No waypoint groups to mapmatch")

	# Takes waypoints from presult['waypoints_used']
	# Extracts a neighborhood graph of the waypoints
	def mapmatch_prepare_subgraph(presult: dict) -> dict :

		# A cloud of candidate edges for each waypoint
		dist_clouds = [dict(waypoints_kne[presult['waypoint_setid']][wp]) for wp in presult['waypoints_used']]

		# The structure of the graph will be modified in this routine
		g1: nx.DiGraph

		if PARAM['do_graph_extract'] :

			# Start with the nodes of nearest edges
			nodes = set(chain.from_iterable(chain.from_iterable(dc.keys()) for dc in dist_clouds))

			waypoints = presult['waypoints_used']

			# Add nodes in a neighborhood of the waypoints
			for (p, x, q) in zip(waypoints, waypoints[1:], waypoints[2:]) :
				r = max(200, 2 * max(commons.geodesic(x, p), commons.geodesic(x, q)))
				nodes |= set(knn['node_ids'][j] for j in knn['knn_tree'].query_radius([x], r=r)[0])

			# Extract subgraph on the 'nodes', keep only the main component
			g1 = g0.subgraph(max(nx.strongly_connected_components(g0.subgraph(nodes)), key=len)).copy()

			# The graph may not contain all the nearest edges anymore
			dist_clouds = [
				{e: d for (e, d) in dc.items() if g1.has_edge(*e)}
				for dc in dist_clouds
			]

			# Remove possible empty edges clouds
			(waypoints, dist_clouds) = zip(*((wp, dc) for (wp, dc) in zip(waypoints, dist_clouds) if dc))
			#
			if (len(waypoints) < len(presult['waypoints_used'])) :
				commons.logger.warning("Number of waypoints reduced from {} to {}".format(len(presult['waypoints_used']), len(waypoints)))
				presult['waypoints_used'] = waypoints

			commons.logger.debug("Mapmatch graph has {} nodes around {} waypoints".format(g1.number_of_nodes(), len(waypoints)))

		else :
			g1 = g0.copy()

		# Edge attr for shortest path computation
		nx.set_edge_attributes(g1, g0_weights, name=PARAM['weight'])

		# Mark the original nodes as basenodes
		nx.set_node_attributes(g1, {n: n for n in g1.nodes}, name='basenode')

		#
		presult['(g)'] = g1
		presult['(dist_clouds)'] = dist_clouds
		presult['(callback)'] = callback
		#
		presult['many_partial'] = many_partial
		presult['stubborn'] = stubborn

		return presult

	# List of 'result' starting points -- "pre-results"
	presults = [
		{**result, **{'waypoint_setid': si, 'waypoints_used': wu}}
		for (si, wu) in groups
	]

	# Mapmatch groups in random order
	if PARAM['waypoints_mapmatch_shuffle'] :
		random.shuffle(presults)

	# Mapmatch only so many groups
	presults = presults[0:PARAM['waypoints_mapmatch_max_groups']]
	assert(presults)

	# MAPMATCH DRIVER
	commons.logger.info("Starting mapmatch on {} subgroups".format(len(presults)))

	def complete_all(presults) -> Generator :

		# Attach a relevant graph extract around the waypoints
		presults = map(mapmatch_prepare_subgraph, presults)

		# The previous step may have reduced the number of waypoints in subgroups
		presults = filter((lambda p : len(p['waypoints_used']) >= PARAM['waypoints_min_number']), presults)

		# Mapmatch -- batch-parallel version
		# Note: 'Parallel' does not yield until all tasks are complete
		for presult_batch in commons.batchup(presults, 5 * PARAM['#jobs']) :
			yield from Parallel(n_jobs=PARAM['#jobs'])(delayed(mapmatch_complete_this)(presult) for presult in presult_batch)

		# # Mapmatch -- serial version
		# for result in incomplete :
		# 	try :
		# 		result = mapmatch_complete_this_2(result)
		# 	except :
		# 		commons.logger.error("Group mapmatch failed within set #{} \n{}".format(result['waypoint_setid'], traceback.format_exc()))
		# 		time.sleep(1)
		# 		continue
		#
		# 	yield result

		# # Mapmatch -- parallel version
		# yield from Parallel(n_jobs=8)(delayed(mapmatch_complete_this_2)(result) for result in incomplete)

	yield from progressbar(complete_all(presults), min_value=1, max_value=len(presults))


# node_pos is a dictionary node id -> (lat, lon)
def compute_geo_knn(node_pos, leaf_size=20, Tree=sklearn.neighbors.BallTree):
	return {
		'node_ids': list(node_pos.keys()),
		'knn_tree': Tree(list(node_pos.values()), leaf_size=leaf_size, metric='pyfunc', func=commons.geodesic)
	}


# Nearest graph edges to a geo-location q = (lat, lon)
def estimate_kne(g, knn, q, ke=10) :
	assert('knn_tree' in knn)
	assert('node_ids' in knn)

	def nearest_nodes(q, kn) :
		# Find nearest nodes using the KNN tree
		(dist, ind) = (boo.reshape(-1) for boo in knn['knn_tree'].query([q], k=kn))
		# Return pairs (graph-node-id, distance-to-q) sorted by distance
		for i in ind :
			yield knn['node_ids'][i]

	# Locate ke nearest directed edges among a set of candidates
	# Return a list of pairs (edge, distance-to-q)
	# Assumes node (lat, lon) coordinates in the node attribute 'pos'
	def filter_nearest_edges(g, q, edges):
		# Identify (a, b) and (b, a)
		iso_edge = (lambda e : (min(e), max(e)))

		# Distance of each (iso)edge to the point q
		ie2d = {
			ie: dist_to_segment(q, (g.nodes[ie[0]]['pos'], g.nodes[ie[1]]['pos']))[0]
			for ie in set(map(iso_edge, edges))
		}

		return sorted([(e, ie2d[iso_edge(e)]) for e in edges], key=(lambda __ : __[1]))[0:ke]

	for kn in range(ke, 20*ke, 5) :
		# Get some closest nodes first
		nn = list(nearest_nodes(q, kn=kn))
		# Get the adjacent edges
		ee = set(g.in_edges(nbunch=nn)) | set(g.out_edges(nbunch=nn))
		# Continue looking for more edges?
		if (len(ee) >= (2*ke)) :
			# Filter down to the nearest ones
			ee = filter_nearest_edges(g, q, ee)
			return ee

	raise RuntimeError("Could not find enough nearest edges")


def test_mapmatch() :

	mapbox_token = open("../.credentials/UV/mapbox-token.txt", 'r').read()

	osm_graph_file = "../OUTPUT/02/UV/kaohsiung.pkl"

	print("Loading OSM...")
	OSM = pickle.load(open(osm_graph_file, 'rb'))

	print("Retrieving the KNN tree...")

	# Road network (main graph component) with nearest-neighbor tree for the nodes
	g : nx.DiGraph
	(g, knn) = commons.inspect(('g', 'knn'))(OSM['main_component_with_knn'])

	kne = (lambda q : estimate_kne(g, knn, q, ke=20))

	# Try to free up memory
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
	waypoints = [(22.60642, 120.338256), (22.60651, 120.338188), (22.606119, 120.338569), (22.606109, 120.33834), (22.605649, 120.333862), (22.60655, 120.33345), (22.60831, 120.333213), (22.617879, 120.332069)]

	# waypoints = [(22.60642, 120.338256), (22.605649, 120.333862), (22.60655, 120.33345), (22.60831, 120.333213), (22.617879, 120.332069), (22.619409, 120.331848), (22.622409, 120.331458), (22.622329, 120.329803), (22.62141, 120.327056), (22.62095, 120.325752), (22.62008, 120.323219), (22.618999, 120.319877), (22.61882, 120.319168), (22.61853, 120.318038), (22.617969, 120.316169), (22.617319, 120.314498), (22.61683, 120.313072), (22.616478999999998, 120.31208), (22.615419, 120.308982), (22.615159, 120.30812), (22.61498, 120.307182), (22.614429, 120.305999), (22.614099, 120.305229), (22.61375, 120.303573), (22.613389, 120.302856), (22.613109, 120.301681), (22.612779, 120.301078), (22.61255, 120.300239), (22.6123, 120.298408), (22.61313, 120.29811), (22.621259, 120.295913), (22.62174, 120.296768), (22.622539, 120.29956), (22.62294, 120.300529), (22.62351, 120.301116), (22.624648, 120.301299), (22.6268, 120.301521)]
	# wp1 = (22.61313, 120.29811) # lower
	# wp2 = (22.621259, 120.295913) # upper
	# print(waypoints.index(wp1), waypoints.index(wp2)) # 29 30
	
	# waypoints = [(22.591286, 120.305706), (22.593253, 120.305906), (22.59474, 120.305306), (22.597746, 120.304186), (22.596886, 120.305133)]

	# waypoints = [(22.613109, 120.301681), (22.612779, 120.301078), (22.61255, 120.300239), (22.6123, 120.298408), (22.61313, 120.29811), (22.621259, 120.295913), (22.62174, 120.296768), (22.622539, 120.29956)]

	# # Off-graph
	# waypoints = [(23.158953, 120.764319), (23.159556, 120.766213), (23.159126, 120.764453), (23.158566, 120.76336), (23.15574, 120.760866), (23.154659, 120.760319), (23.151839, 120.757533), (23.14992, 120.75664), (23.14823, 120.755399), (23.146363, 120.754159), (23.145453, 120.750733), (23.144263, 120.749773), (23.138233, 120.738666), (23.137466, 120.7296), (23.13557, 120.724706), (23.134416, 120.723146), (23.133646, 120.72244), (23.132079, 120.720439), (23.130616, 120.71916), (23.12967, 120.717573), (23.129436, 120.71688), (23.127803, 120.714626), (23.119859, 120.710013), (23.116743, 120.712706), (23.11632, 120.713466), (23.114913, 120.713279), (23.110603, 120.712013), (23.109433, 120.710333), (23.110526, 120.7086), (23.108769, 120.699866), (23.105703, 120.694919), (23.103096, 120.690733), (23.101586, 120.688879), (23.100159, 120.685826), (23.095286, 120.682013), (23.09337, 120.681706), (23.08988, 120.682866), (23.08783, 120.681039), (23.086396, 120.679666), (23.08473, 120.679426), (23.079733, 120.676773), (23.079039, 120.676333), (23.077429, 120.675373), (23.073783, 120.673479), (23.073319, 120.673133), (23.072413, 120.672933), (23.06987, 120.672719), (23.067569, 120.673386), (23.06427, 120.672773), (23.064096, 120.671453), (23.062183, 120.670893), (23.052396, 120.667506), (23.049193, 120.669039), (23.046186, 120.667866), (23.042326, 120.667413), (23.03734, 120.665693), (23.025093, 120.663466), (23.020016, 120.664373), (23.015033, 120.665293), (23.012633, 120.665346), (23.007923, 120.664519), (23.00765, 120.662826), (23.006326, 120.654946), (23.006913, 120.651799), (23.006509, 120.649106), (23.004886, 120.647413), (23.003593, 120.646879), (22.999726, 120.644453), (22.996513, 120.643013), (22.995793, 120.642453), (22.996606, 120.641493), (22.996259, 120.64068), (22.995889, 120.634906), (22.996983, 120.63436), (22.99773, 120.634213)]

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
			try :
				mapi = maps.get_map_by_bbox(maps.ax2mb(*ax.axis()), token=mapbox_token)
			except maps.URLError :
				commons.logger.warning("No map (no connection?)")
				mapi = None

			result['plt'] = { 'fig' : fig, 'ax' : ax, 'map' : mapi, 'bbox' : ax.axis() }

		if (result['status'] in ["zero", "opti", "done"]) :

			fig : plt.Figure
			ax : plt.Axes
			(fig, ax, mapi, bbox) = commons.inspect({'plt': ('fig', 'ax', 'map', 'bbox')})(result)

			# Clear the axes
			ax.cla()

			# Apply the background map
			if mapi :
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
			open("graph_mm_callback.gpx", 'w').write(simple_gpx(result['waypoints'], [result['geo_path']]).to_xml())

	print("Calling mapmatch...")

	plt.ion()

	commons.seed()

	for _ in range(2) :
		try :
			result = mapmatch(waypoints, g, kne, mm_callback, stubborn=0.2)
			print(result['path'])
		except MapMatchingError as e :
			print("Mapmatch failed ({})".format(e))

		plt.pause(5)

	plt.ioff()
	plt.show()

	return


if (__name__ == "__main__") :
	test_mapmatch()
	# test_dist_to_segment()
