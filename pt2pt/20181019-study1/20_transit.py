#!/usr/bin/python3

# RA, 2018-10-21

## ================== IMPORTS :

from helpers import commons, graph, transit, maps

import os
import pickle
import inspect
import datetime as dt
import numpy as np
import pandas as pd
import networkx as nx
import sklearn.neighbors

from copy import deepcopy
from joblib import Parallel, delayed
from progressbar import progressbar
from itertools import chain

from collections import defaultdict

import pytz

## ==================== NOTES :

pass


## ===================== META :

# What finally identifies a one-way route
ROUTE_KEY = (lambda r : (r['SubRouteUID'], r['Direction']))


## ================== PARAM 1 :

PARAM = {
	'city' : "Kaohsiung",
	'scenario' : "Kaohsiung/20181105-20181111",
	'TZ' : pytz.timezone('Asia/Taipei'),
}


## ==================== INPUT :

IFILE = {
	#'OSM-pickled' : "OUTPUT/02/UV/kaohsiung.pkl",

	# Will be loaded from timetable files:
	#'MOTC_routes' : "OUTPUT/00/ORIGINAL_MOTC/{city}/CityBusApi_StopOfRoute.json",
	#'MOTC_stops'  : "OUTPUT/00/ORIGINAL_MOTC/{city}/CityBusApi_Stop.json",

	'MOTC_shapes' : "OUTPUT/00/ORIGINAL_MOTC/{city}/CityBusApi_Shape.json",

	'timetable_json' : "OUTPUT/17/timetable/{scenario}/json/{{routeid}}-{{dir}}.json",
}

for (k, s) in IFILE.items() : IFILE[k] = s.format(**PARAM)


## =================== OUTPUT :

OFILE = {
	'' : "",
}


## ================== PARAM 2 :

PARAM.update({
	'walker_neighborhood_radius' : 250, # meters
	'walker_speed' : 1, # m/s
	'walker_delay' : 5, # seconds
})


## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))


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

	def where_can_i_go(self, P: transit.Loc, tspan=None) :
		try :
			# See if geo-coordinate is available
			(lat, lon) = P.x
		except :
			# Interpret P.desc as a bus stop; get its geo-coordinate
			P.x = self.stop_pos[P.desc]

		transit_info = {
			B : transit.Leg(
				P,
				transit.Loc(
					t=(P.t + dt.timedelta(seconds=(PARAM['walker_delay'] + commons.geodesic(P.x, self.stop_pos[B]) / PARAM['walker_speed']))),
					x=self.stop_pos[B],
					desc=B
				),
				mode=transit.Mode.walk
			)
			for B in self.get_neighbors(P.x)
		}

		return transit_info

	def __del__(self) :
		pass


class BusstopBusser :
	def __init__(self, routes, stop_pos) :
		self.routes = { ROUTE_KEY(route) : route for route in routes }
		self.routes_from = defaultdict(set)
		for (i, route) in self.routes.items() :
			assert(type(route['Direction']) is int), "Routes should be unidirectional at this point"
			# Note: we are excluding the last stop (no boarding there)
			for stop in route['Stops'][:-1] :
				self.routes_from[stop['StopUID']].add(i)
		self.routes_from = dict(self.routes_from)
		#print(list(self.routes_from.items())[0:10])

		self.stop_pos = stop_pos

	def where_can_i_go(self, P: transit.Loc, tspan=dt.timedelta(minutes=60)) :
		# Stop ID --> Transit leg
		dest = dict()
		P = deepcopy(P)

		# Location descriptor (usually, bus stop ID)
		A = P.desc

		try :
			routes = self.routes_from[A]
		except :
			# 'stopid' is not recognized as a bus stop ID
			return dest

		for route_key in routes :

			fn = IFILE['timetable_json'].format(**dict(zip(['routeid', 'dir'], route_key)))
			if not os.path.isfile(fn) :
				print("Warning: No timetable for route {}".format(route_key))
				continue

			# Note: 'datetime64[ms]' appears necessary here to properly parse the JSON
			#       but the resulting datatype is '...[ns]'
			tt : pd.DataFrame # Timetable
			tt = pd.read_json(commons.zipjson_load(fn)['timetable_df'], dtype='datetime64[ms]').dropna()
			tt = tt.astype(pd.Timestamp)

			# Find the reachable section of the time table
			tt = tt.loc[(P.t <= tt[A]) & (tt[A] <= (P.t + tspan)), A:]

			if tt.empty :
				continue

			# This should not happen, but just in case
			if (len(tt.columns) < 2) :
				continue

			# Next stop name
			B = tt.columns[1]

			# Note: Convert to numpy datetime for the argmin computation
			fastest: pd.Series
			fastest = tt.ix[tt[B].astype(np.datetime64).idxmin()]

			P.t = fastest[A].to_pydatetime()
			Q = transit.Loc(t=fastest[B].to_pydatetime(), x=self.stop_pos[B], desc=B)

			leg = transit.Leg(P, Q, transit.Mode.bus, desc={'route': route_key, 'bus_id': fastest.name})

			if B in dest :
				dest[B] = min(dest[B], leg, key=(lambda _: _.Q.t))
			else :
				dest[B] = leg

		return dest

## ==================== TESTS :

def test1() :
	t0 = PARAM['TZ'].localize(dt.datetime(year=2018, month=11, day=6, hour=13, minute=15))
	print("Departure time: {}".format(t0.strftime("%Y-%m-%d %H:%M (%Z)")))

	#
	#t0 = t0.astimezone(dt.timezone.utc).replace(tzinfo=None)

	def ll2xy(latlon) :
		return (latlon[1], latlon[0])

	routes = [
		commons.zipjson_load(fn)['route']
		for fn in commons.ls(IFILE['timetable_json'].format(routeid="*", dir="*"))
	]

	stops = {
		stop['StopUID'] : stop
		for route in routes
		for stop in route['Stops']
	}

	stop_pos = {
		stopid : commons.inspect({'StopPosition' : ('PositionLat', 'PositionLon')})(stop)
		for (stopid, stop) in stops.items()
	}


	bb = BusstopBusser(routes, stop_pos)

	# print(bb.where_can_i_go('KHH308', t))
	# exit(39)

	# print("Routes passing through {}: {}".format(A, bb.routes_from[A]))
	# print(bb.routes[('KHH100', 0)]['Stops'])
	# exit(39)

	bw = BusstopWalker(stop_pos)
	# print(bw.where_can_i_go('KHH308', t))
	# bw.where_can_i_go('KHH380', t)


	(A, Z) = commons.random_subset(stop_pos.keys(), k=2)
	(A, Z) = ('KHH4439', 'KHH4370')

	# # Long search, retakes same busroute
	# (A, Z) = ('KHH3820', 'KHH4484')

	# Initial openset -- start locations
	aa = { transit.Loc(t=t0.astimezone(dt.timezone.utc).replace(tzinfo=None), x=stop_pos[A], desc=A) } # 'KHH4560', 'KHH308', 'KHH12822'
	# Target locations
	zz = { transit.Loc(t=None, x=stop_pos[Z], desc=Z) } # 'KHH4391'

	# Relatively short route, four buses

	print("Finding a route from {} to {}".format(aa, zz))



	import matplotlib.pyplot as plt
	plt.ion()
	(fig, ax) = plt.subplots()


	g = nx.DiGraph()

	# A*-algorithm heuristic: cost estimate from C to Z
	# It is "admissible" if it never over-estimates
	def h(P: transit.Loc) :
		return min(
			dt.timedelta(seconds=(commons.geodesic(P.x, Q.x) / (3 * PARAM['walker_speed'])))
			for Q in zz
		)

	# A*-algorithm cost estimator of path via C
	def f(P: transit.Loc) :
		return P.t + h(P)

	openset = set(aa)

	for P in openset :
		g.add_node(P.desc, t=(P.t).astimezone(dt.timezone.utc).replace(tzinfo=None), pos=ll2xy(P.x))

	# Next figure update
	nfu = dt.datetime.now()

	while openset :

		# A*-algorithm: select candidate node/path to extend
		c = min(openset, key=f)

		openset.remove(c)


		leg_choices = chain.from_iterable(
			mode.where_can_i_go(c).values()
			for mode in [bb, bw]
		)

		leg: transit.Leg
		for leg in leg_choices :

			# Next potential bus stop
			B = leg.Q.desc

			if B in g.nodes :
				if (g.nodes[B]['t'] <= leg.Q.t) :
					continue
				else :
					g.remove_edges_from(list(g.in_edges(B)))

			g.add_node(leg.Q.desc, t=leg.Q.t, pos=ll2xy(leg.Q.x), leg=deepcopy(leg))
			g.add_edge(leg.P.desc, leg.Q.desc, mode=leg.mode)
			openset.add(leg.Q)

			#print("Added new path to:", B)

			if B in [P.desc for P in zz] :
				#print("Done!")
				openset.clear()

				# Retrace path
				legs = []
				A = B
				while (A not in [P.desc for P in aa]) :
					legs.append(deepcopy(g.nodes[A]['leg']))
					A = legs[-1].P.desc
				legs.reverse()

				for leg in legs :
					(t0, t1) = (pytz.utc.localize(t).astimezone(tz=PARAM['TZ']) for t in (leg.P.t, leg.Q.t))
					print("{}-{} : {} {}".format(t0.strftime("%Y-%m-%d %H:%M"), t1.strftime("%H:%M (%Z)"), leg.mode, leg.desc))

				break

		if (dt.datetime.now() >= nfu) or (not openset) :
			ax: plt.Axes
			ax.cla()

			nx.draw_networkx_edges(g, ax=ax, edgelist=[(a, b) for (a, b, d) in g.edges.data('mode') if (d == transit.Mode.walk)], pos=nx.get_node_attributes(g, 'pos'), edge_color='g', arrowsize=5, node_size=0)
			nx.draw_networkx_edges(g, ax=ax, edgelist=[(a, b) for (a, b, d) in g.edges.data('mode') if (d == transit.Mode.bus )], pos=nx.get_node_attributes(g, 'pos'), edge_color='b', arrowsize=5, node_size=0)
			if aa :
				ax.plot(*zip(*[ll2xy(P.x) for P in aa]), 'go')
			if zz :
				ax.plot(*zip(*[ll2xy(P.x) for P in zz]), 'ro')
			if openset :
				ax.plot(*zip(*[ll2xy(O.x) for O in openset]), 'kx')

			if not openset :
				try :
					for leg in legs :
						(y, x) = zip(leg.P.x, leg.Q.x)
						ax.plot(x, y, 'y-', alpha=0.3, linewidth=8, zorder=100)
				except :
					pass

			a = ax.axis()
			for route in routes :
				(y, x) = zip(*(stop_pos[stop['StopUID']] for stop in route['Stops']))
				ax.plot(x, y, 'm-', alpha=0.1)
			ax.axis(a)

			plt.pause(0.1)

			nfu = dt.datetime.now() + dt.timedelta(seconds=2)

	print("Done.")

	plt.ioff()
	plt.show()



## ===================== WORK :


## ==================== ENTRY :

if (__name__ == "__main__") :
	test1()

