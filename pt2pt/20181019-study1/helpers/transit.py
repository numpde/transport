#!/usr/bin/python3

# RA, 2018-12-16

## ================== IMPORTS :

import datetime as dt


import numpy as np
import pandas as pd
import networkx as nx

from heapdict import heapdict

import sklearn.neighbors

from copy import deepcopy
from enum import Enum

from collections import defaultdict
from itertools import chain, groupby

# from joblib import Parallel, delayed
# from progressbar import progressbar

from helpers import commons, graph


## ==================== PARAM :

PARAM = {
	# Consider bus stops within this radius
	'busstop_walker_cutoff_radius' : 50, # meters
	# Average walker speed
	'busstop_walker_speed' : 1, # m/s

	# Number of graph neighbors to consider from an arbitrary location
	'graph_walker_#neighbors' : 10,
	# Average walker speed
	'graph_walker_speed' : 1, # m/s

	'prefilter_legs' : False,
}

## ===================== META :

# What finally identifies a one-way route
ROUTE_KEY = (lambda r : (r['SubRouteUID'], r['Direction']))


## ====================== AUX :

def ll2xy(latlon) :
	return (latlon[1], latlon[0])

def get_neighbors(knn, x, radius=None, k=None) :
	tree : sklearn.neighbors.BallTree
	(I, tree) = commons.inspect(('node_ids', 'knn_tree'))(knn)
	if radius :
		return [I[j] for j in tree.query_radius([x], r=radius)[0]]
	if k :
		return [I[j] for j in tree.query([x], k=k, return_distance=False)[0]]
	raise ValueError("Either *radius* or *number of neighbors* should be provided")

## ========== BASIC DATATYPES :

class Loc :
	# Expect:
	# 't' to be a timepoint
	# 'x' to be a (lat, lon) pair
	# 'desc' some descriptor of the location
	def __init__(self, t=None, x=None, desc=None) :
		t: dt.datetime
		if t and t.tzinfo :
			t = t.astimezone(dt.timezone.utc).replace(tzinfo=None)
		self.t = t
		self.x = x
		self.desc = desc
	def __str__(self) :
		return "{}/{} at {}".format(self.desc, self.x, self.t)
	def __repr__(self) :
		return "Loc(t={}, x={}, desc={})".format(self.t, self.x, self.desc)


class Mode(Enum) :
	walk = "Walk"
	bus = "Bus"


class Leg :
	def __init__(self, P: Loc, Q: Loc, mode: Mode, desc=None) :
		self.P = P
		self.Q = Q
		self.mode = mode
		self.desc = desc
	def __str__(self) :
		return "({P})--[{mode}/{desc}]-->({Q})".format(P=self.P, Q=self.Q, mode=self.mode, desc=self.desc)


## =================== SLAVES :

class GraphWalker :
	def __init__(self, g: nx.Graph, knn, node_pos) :
		self.g = g
		self.knn = knn
		self.node_pos = node_pos

	def where_can_i_go(self, O: Loc) :
		try :
			# Interpret the origin as a node of the graph
			neighbors = self.g.neighbors(O.desc)
		except :
			# Just locate the nearest node(s) of the graph
			neighbors = get_neighbors(self.knn, O.x, k=PARAM['graph_walker_#neighbors'])

		for nodeid in neighbors :
			# Encode a walk
			yield Leg(
				Loc(t=O.t, x=O.x, desc=O.desc),
				Loc(
					t=(O.t + dt.timedelta(seconds=(commons.geodesic(O.x, self.node_pos[nodeid]) / PARAM['graph_walker_speed']))),
					x=self.node_pos[nodeid],
					desc=nodeid
				),
				mode=Mode.walk
			)


class BusstopWalker :

	def __init__(self, stop_pos) :
		self.stop_pos = deepcopy(stop_pos)
		self.knn = graph.compute_geo_knn(self.stop_pos)

	def where_can_i_go(self, O: Loc, tspan=None) :
		try :
			# See if geo-coordinate is available
			(lat, lon) = O.x
		except :
			# Interpret O.desc as a bus stop; get its geo-coordinate
			O.x = self.stop_pos[O.desc]
			# Actually, do not expect this to happen
			commons.logger.debug("No geo-coordinate in BusstopWalker")

		# "Orientation" delay before start walking
		t = O.t # + dt.timedelta(seconds=PARAM['walker_delay'])

		for B in get_neighbors(self.knn, O.x, radius=PARAM['busstop_walker_cutoff_radius']) :
			# Encode a walk
			yield Leg(
				Loc(t=t, x=O.x, desc=O.desc),
				Loc(
					t=(t + dt.timedelta(seconds=(commons.geodesic(O.x, self.stop_pos[B]) / PARAM['busstop_walker_speed']))),
					x=self.stop_pos[B],
					desc=B
				),
				mode=Mode.walk
			)


class BusstopBusser :
	def __init__(self, routes, stop_pos, timetables) :

		self.routes_from = defaultdict(set)
		#
		for (i, route) in routes.items() :
			assert(type(route['Direction']) is int), "Routes should be unidirectional at this point"
			# Note: we are excluding the last stop of the route (no boarding there)
			for stop in route['Stops'][:-1] :
				self.routes_from[stop['StopUID']].add(i)
		#
		self.routes_from = dict(self.routes_from)

		self.stop_pos = stop_pos
		self.timetables = timetables

		with commons.Timer('patch_timetable_monotonicity') :
			self.patch_timetable_monotonicity()

	def patch_timetable_monotonicity(self) :
		self.timetables = { k: tt.apply(np.maximum.accumulate, axis=1).astype(pd.Timestamp) for (k, tt) in self.timetables.items() }


	def where_can_i_go(self, O: Loc, tspan=dt.timedelta(minutes=60)) :

		# Earliest time of departure and location descriptor (usually, bus stop ID)
		(t, this_stop) = (O.t, O.desc)

		try :
			routes_through_here = self.routes_from[this_stop]
		except :
			# 'this_stop' is not recognized as a bus stop ID
			return []

		# Transit legs; may contain repeated destinations
		legs = []

		for route_key in routes_through_here :
			tt : pd.DataFrame # Timetable
			tt = self.timetables[route_key]

			# commons.logger.debug(O.t, tt[this_stop])

			# Find the reachable section of the time table
			reachable = tt.loc[(t <= tt[this_stop]) & (tt[this_stop] <= (t + tspan)), this_stop:]

			# No connection at this space-time point
			if reachable.empty : continue

			# This should not happen, but just in case
			if (len(reachable.columns) < 2) : continue

			# Next stop name
			next_stop = reachable.columns[1]

			# Note: Convert to numpy datetime for the argmin computation
			fastest: pd.Series
			fastest = reachable.ix[reachable[next_stop].astype(np.datetime64).idxmin()]

			P = Loc(t=fastest[this_stop].to_pydatetime(), x=O.x, desc=O.desc)
			Q = Loc(t=fastest[next_stop].to_pydatetime(), x=self.stop_pos[next_stop], desc=next_stop)

			leg = Leg(P, Q, Mode.bus, desc={'route': route_key, 'bus_id': fastest.name})

			if not (leg.P.t <= leg.Q.t) :
				commons.logger.warning("Non-monotonic timedelta in leg of route {}".format(route_key))

			yield leg


## =================== MASTER :

class Transit :
	def __init__(self, timetable_files, graph_with_knn) :
		#
		timetable_files = list(timetable_files)

		self.routes = {
			ROUTE_KEY(tt['route']) : tt['route']
			for tt in map(commons.zipjson_load, timetable_files)
		}

		# Note: 'datetime64[ms]' appears necessary here to properly parse the JSON
		#       but the resulting datatype is '...[ns]'
		self.timetables = {
			ROUTE_KEY(tt['route']) : pd.read_json(tt['timetable_df'], dtype='datetime64[ms]').dropna().astype(pd.Timestamp)
			for tt in map(commons.zipjson_load, timetable_files)
		}

		self.stop_pos = {
			stop['StopUID'] : commons.inspect({'StopPosition' : ('PositionLat', 'PositionLon')})(stop)
			for route in self.routes.values()
			for stop in route['Stops']
		}

		(g, knn) = commons.inspect(('g', 'knn'))(graph_with_knn)
		self.node_pos = nx.get_node_attributes(g, 'pos')

		self.bb = BusstopBusser(self.routes, self.stop_pos, self.timetables)
		self.bw = BusstopWalker(self.stop_pos)
		self.gw = GraphWalker(g, knn, self.node_pos)

	def completed_loc(self, loc: Loc) :
		try :
			# See if 'x' is available
			(lat, lon) = loc.x
			return deepcopy(loc)
		except :
			# Try to interpret 'desc' as a bus stop; get its geo-coordinate
			return deepcopy(Loc(t=loc.t, x=self.stop_pos[loc.desc], desc=loc.desc))

	def now(self) :
		return dt.datetime.now().astimezone().isoformat()

	# Run A*-algorithm to connect 'loc_a' and 'loc_b'
	def connect(self, loc_a: Loc, loc_b=None, callback=None) :
		result = { 'status' : "zero", 'time_start' : self.now() }
		if callback : callback(result)

		if loc_a :
			try :
				# Initial astar_openset -- start locations
				loc_a = self.completed_loc(loc_a)
				astar_initial = {loc_a.desc : loc_a}
			except :
				raise ValueError("Start location not understood")
		else :
			raise ValueError("No start location provided")

		if loc_b :
			try :
				# Target locations
				loc_b = self.completed_loc(loc_b)
				astar_targets = {loc_b.desc : loc_b}
			except :
				raise ValueError("Target location not understood")
		else :
			astar_targets = {}


		# A*-algorithm heuristic: cost estimate from P to Targets
		# It is "admissible" if it never over-estimates
		def astar_cost_h(P: Loc) :
			if astar_targets :
				return min(
					dt.timedelta(seconds=(commons.geodesic(P.x, Q.x) / (3 * PARAM['walker_speed'])))
					for Q in astar_targets.values()
				)
			else :
				return dt.timedelta(seconds=0)

		# A*-algorithm cost of path to P
		def astar_cost_g(P: Loc) :
			return P.t

		# A*-algorithm cost estimator of path via P
		def astar_cost_f(P: Loc) :
			return astar_cost_g(P) + astar_cost_h(P)


		# Working copy of the open set
		astar_openset = deepcopy(astar_initial)
		# Keep track of the cost for efficient retrieval of min-cost element
		astar_openset_priority = heapdict({ k : astar_cost_f(P) for (k, P) in astar_openset.items() })
		astar_openset_priority.popit = (lambda : astar_openset_priority.popitem()[0])

		# The graph of visited locations will act as the closed set
		astar_graph = nx.DiGraph()

		# Initialize the graph
		for (desc, P) in astar_openset.items() :
			astar_graph.add_node(desc, loc=P)


		# These do not change during the main loop
		result['routes'] = deepcopy(self.routes)
		result['stop_pos'] = deepcopy(self.stop_pos)
		result['astar_initial'] = deepcopy(astar_initial)
		result['astar_targets'] = deepcopy(astar_targets)

		# These do change during the main loop
		def do_callback(status) :
			if not callback : return

			# No deepcopy here for performance reasons
			result['astar_openset'] = (astar_openset)
			result['astar_graph'] = (astar_graph)

			result['status'] = status
			result['time_finish'] = self.now()

			callback(result)

		do_callback("init")


		# MAIN A*-ALGORITHM LOOP
		while astar_openset :

			# commons.logger.debug("Open set: {}".format(sorted(astar_openset)))

			# A*-algorithm: select candidate node/path with smallest cost
			# https://github.com/DanielStutzbach/heapdict
			C: Loc
			C = astar_openset.pop(astar_openset_priority.popit())

			# Collect potential next moves
			leg_choices = set()
			# From bus stop by bus to some other bus stop
			with commons.Timer('astar_collect_choices_bb') :
				leg_choices |= set(self.bb.where_can_i_go(C))
			# From any location to nearby bus stop
			with commons.Timer('astar_collect_choices_bw') :
				leg_choices |= set(self.bw.where_can_i_go(C))
			# From any location to nearby graph nodes, or from node to node along edge
			with commons.Timer('astar_collect_choices_gw') :
				leg_choices |= set(self.gw.where_can_i_go(C))

			if PARAM['prefilter_legs'] :
				# Keep only the most efficient candidate for each next destination
				leg_choices = {
					min(legs, key=(lambda __ : astar_cost_g(__.Q)))
					# Sort and group by 'desc', i.e. the bus stop ID
					for (k, legs) in commons.sort_and_group(
						leg_choices, key=(lambda __ : __.Q.desc)
					)
				}

			# All next moves should originate with the selected location
			assert({C.desc} == {leg.P.desc for leg in leg_choices})

			# All next moves should respect monotonicity of the cost
			assert(all((astar_cost_g(C) <= astar_cost_g(leg.P)) for leg in leg_choices))

			leg: Leg
			for leg in leg_choices :

				with commons.Timer('astar_process_choices') :

					# Bus stop we are coming from & may be going to
					(prev_stop, next_stop) = (leg.P.desc, leg.Q.desc)

					# Note: prev_stop may be None if it is the origin of search, unlike next_stop
					assert(next_stop is not None), "Next stop should have an ID"

					# Have we visited that stop already?
					if astar_graph.has_node(next_stop) :
						# Compare the path cost this far
						if (astar_cost_g(astar_graph.nodes[next_stop]['loc']) <= astar_cost_g(leg.Q)) :
							# This move offers no improvement in cost
							continue
						else :
							# Remove incoming edges to be replaced
							# Maintains the integrity of the transport graph throughout, unlike 'remove_node'
							astar_graph.remove_edges_from(set(astar_graph.in_edges(next_stop)))

					# Make an edge from 'prev_stop' to 'next_stop'
					astar_graph.add_edge(prev_stop, next_stop, leg=leg)
					# Update node attributes
					astar_graph.add_node(next_stop, loc=leg.Q)

					try :
						# If 'next_stop' is in the open set already, keep only the better suggestion
						astar_openset[next_stop] = min(astar_openset[next_stop], leg.Q, key=astar_cost_g)
					except KeyError :
						# Assume 'next_stop' is not in the open set
						astar_openset[next_stop] = leg.Q
					finally :
						# Priority of 'next_stop' for subsequent selection (lower = earlier)
						astar_openset_priority[next_stop] = astar_cost_f(astar_openset[next_stop])


				# A* TERMINATION CRITERION
				# If reached the target retrace the path
				if next_stop in astar_targets :

					def retrace_from_node(last_stop) :
						while astar_graph.pred[last_stop] :
							assert (1 == len(astar_graph.pred)), "By construction, a node should be reached through one path only"
							(last_stop, edge_data) = next(iter(astar_graph.pred[last_stop].items()))
							yield edge_data['leg']

					# Retrace
					legs = reversed(list(retrace_from_node(next_stop)))

					# Group by transport mode/vehicle
					legs = [
						list(group)
						for (k, group) in groupby(legs, key=(lambda leg: (leg.mode, leg.desc)))
					]

					# Collapse groups
					legs = [
						Leg(P=group[0].P, Q=group[-1].Q, mode=group[0].mode, desc=group[0].desc)
						for group in legs
					]

					result['legs'] = deepcopy(legs)
					do_callback("done")

					return legs

			for leg in leg_choices :
				assert(leg.Q.desc in astar_graph.nodes)
				assert(astar_cost_g(astar_graph.nodes[leg.Q.desc]['loc']) <= astar_cost_g(leg.Q))

			do_callback("opti")

		if astar_targets :
			# The function should have *return*ed by this point
			do_callback("fail")
			raise RuntimeError("No connection found")
		else:
			do_callback("done")
