#!/usr/bin/python3

# RA, 2018-12-16

## ================== IMPORTS :

import datetime as dt


import os

from itertools import chain, groupby

import numpy as np
import pandas as pd
import networkx as nx

import sklearn.neighbors

from helpers import commons, graph
from copy import deepcopy
from enum import Enum

from collections import defaultdict

from joblib import Parallel, delayed
from progressbar import progressbar


## ==================== PARAM :

PARAM = {
	'walker_neighborhood_radius' : 500, # meters
	'walker_speed' : 1, # m/s
	'walker_delay' : 5, # seconds
}

## ===================== META :

# What finally identifies a one-way route
ROUTE_KEY = (lambda r : (r['SubRouteUID'], r['Direction']))


## ====================== AUX :

def ll2xy(latlon) :
	return (latlon[1], latlon[0])


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
		#return "<Location '{}' at {}>".format(self.desc, self.x)
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

class BusstopWalker :

	def __init__(self, stop_pos) :
		self.stop_pos = deepcopy(stop_pos)
		self.knn = graph.compute_geo_knn(self.stop_pos)

	def get_neighbors(self, x) :
		try :
			(lat, lon) = x
		except :
			raise ValueError("Expect a (lat, lon) geo-coordinate")

		tree : sklearn.neighbors.BallTree
		(I, tree) = commons.inspect(('node_ids', 'knn_tree'))(self.knn)
		return [I[j] for j in tree.query_radius([x], r=PARAM['walker_neighborhood_radius'])[0]]

	def where_can_i_go(self, P: Loc, tspan=None) :
		try :
			# See if geo-coordinate is available
			(lat, lon) = P.x
		except :
			# Interpret P.desc as a bus stop; get its geo-coordinate
			P.x = self.stop_pos[P.desc]

		dest = {
			B : Leg(
				P,
				Loc(
					t=(P.t + dt.timedelta(seconds=(PARAM['walker_delay'] + commons.geodesic(P.x, self.stop_pos[B]) / PARAM['walker_speed']))),
					x=self.stop_pos[B],
					desc=B
				),
				mode=Mode.walk
			)
			for B in self.get_neighbors(P.x)
		}

		return list(dest.values())

	def __del__(self) :
		pass


class BusstopBusser :
	def __init__(self, routes, stop_pos, timetables) :

		self.routes_from = defaultdict(set)

		for (i, route) in routes.items() :
			assert(type(route['Direction']) is int), "Routes should be unidirectional at this point"
			# Note: we are excluding the last stop (no boarding there)
			for stop in route['Stops'][:-1] :
				self.routes_from[stop['StopUID']].add(i)

		self.routes_from = dict(self.routes_from)
		#print(list(self.routes_from.items())[0:10])

		self.stop_pos = stop_pos
		self.timetables = timetables

	def where_can_i_go(self, P: Loc, tspan=dt.timedelta(minutes=60)) :
		P = deepcopy(P)

		# Location descriptor (usually, bus stop ID)
		A = P.desc

		try :
			routes_through_here = self.routes_from[A]
		except :
			# 'A' is not recognized as a bus stop ID
			return []

		# Stop ID --> Transit leg
		dest = dict()

		for route_key in routes_through_here :
			tt : pd.DataFrame # Timetable
			tt = self.timetables[route_key]

			# Find the reachable section of the time table
			reachable = tt.loc[(P.t <= tt[A]) & (tt[A] <= (P.t + tspan)), A:]

			# No connection at this space-time point
			if reachable.empty : continue

			# This should not happen, but just in case
			if (len(reachable.columns) < 2) : continue

			# Next stop name
			next_stop = reachable.columns[1]

			# Note: Convert to numpy datetime for the argmin computation
			fastest: pd.Series
			fastest = reachable.ix[reachable[next_stop].astype(np.datetime64).idxmin()]

			P.t = fastest[A].to_pydatetime()
			Q = Loc(t=fastest[next_stop].to_pydatetime(), x=self.stop_pos[next_stop], desc=next_stop)

			leg = Leg(P, Q, Mode.bus, desc={'route': route_key, 'bus_id': fastest.name})

			if next_stop in dest :
				dest[next_stop] = min(dest[next_stop], leg, key=(lambda _ : _.Q.t))
			else :
				dest[next_stop] = leg

		return list(dest.values())


## =================== MASTER :

class Transit :
	def __init__(self, timetable_files) :
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

		self.bb = BusstopBusser(self.routes, self.stop_pos, self.timetables)
		self.bw = BusstopWalker(self.stop_pos)

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

	def connect(self, loc_a: Loc, loc_b=None, callback=None) :
		result = { 'status' : "zero", 'time_start' : self.now() }
		if callback : callback(result)

		if True :
			try :
				# Initial astar_openset -- start locations
				loc_a = self.completed_loc(loc_a)
				astar_initial = {loc_a.desc : loc_a}
			except :
				raise ValueError("Start location not understood")

		if loc_b :
			try :
				# Target locations
				loc_b = self.completed_loc(loc_b)
				astar_targets = {loc_b.desc : loc_b}
			except :
				raise ValueError("Target location not understood")
		else :
			astar_targets = {}

		# Working copy of the open set
		astar_openset = deepcopy(astar_initial)

		# The graph of visited locations will act as the closed set
		astar_graph = nx.DiGraph()

		# Initialize the graph
		for (desc, P) in astar_openset.items() :
			astar_graph.add_node(desc, pos=ll2xy(P.x), P=P)


		# A*-algorithm heuristic: cost estimate from C to Z
		# It is "admissible" if it never over-estimates
		def h(P: Loc) :
			if astar_targets :
				return min(
					dt.timedelta(seconds=(commons.geodesic(P.x, Q.x) / (3 * PARAM['walker_speed'])))
					for Q in astar_targets.values()
				)
			else :
				return dt.timedelta(seconds=0)

		# A*-algorithm cost estimator of path via C
		def f(P: Loc) :
			return P.t + h(P)


		result['routes'] = deepcopy(self.routes)
		result['stop_pos'] = deepcopy(self.stop_pos)
		result['astar_initial'] = deepcopy(astar_initial)
		result['astar_targets'] = deepcopy(astar_targets)

		def do_callback(status) :
			if not callback : return
			result['astar_openset'] = (astar_openset)
			result['astar_graph'] = (astar_graph)
			result['status'] = status
			result['time_finish'] = self.now()
			callback(result)

		do_callback("init")

		# Main A*-algorithm loop
		while astar_openset :

			# A*-algorithm: select candidate node/path to extend
			c = astar_openset.pop(min(astar_openset.values(), key=f).desc)


			leg_choices = chain.from_iterable(
				mode.where_can_i_go(c)
				for mode in [self.bb, self.bw]
			)

			leg: Leg
			for leg in leg_choices :

				# Bus stop we are coming from & maybe going to
				(prev_stop, next_stop) = (leg.P.desc, leg.Q.desc)

				# Have we visited that stop already?
				if next_stop in astar_graph.nodes :
					if (astar_graph.nodes[next_stop]['P'].t <= leg.Q.t) :
						# No improvement
						continue
					else :
						astar_graph.remove_node(next_stop)

				astar_graph.add_node(next_stop, pos=ll2xy(leg.Q.x), P=leg.Q)
				astar_graph.add_edge(prev_stop, next_stop, leg=leg)

				astar_openset[next_stop] = leg.Q

				#print("Added new path to:", B)

				if next_stop in astar_targets :
					#print("Done!")
					astar_openset.clear()

					def retrace_from_node(a) :
						while astar_graph.in_edges(a) :
							(a, b) = (next(iter(astar_graph.predecessors(a))), a)
							yield astar_graph.edges[a, b]['leg']

					legs = [
						list(group)
						for (k, group) in groupby(reversed(list(retrace_from_node(next_stop))), key=(lambda leg: (leg.mode, leg.desc)))
					]

					legs = [
						Leg(P=group[0].P, Q=group[-1].Q, mode=group[0].mode, desc=group[0].desc)
						for group in legs
					]

					result['legs'] = deepcopy(legs)
					do_callback("done")

					return legs

			do_callback("opti")

		if astar_targets :
			do_callback("fail")
			raise RuntimeError("No connection found")
		else:
			do_callback("done")
