#!/usr/bin/python3

# RA, 2018-10-21

## ================== IMPORTS :

from helpers import commons, graph, maps

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
	'walker_max_busstop_distance' : 250, # meters
	'walker_speed' : 1, # m/s
})


## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))


## =================== SLAVES :

class BusstopWalker :

	def __init__(self, stop_pos) :
		self.stop_pos = deepcopy(stop_pos)
		self.knn = graph.compute_geo_knn(self.stop_pos)

	def get_neighbors(self, stopid) :
		tree : sklearn.neighbors.BallTree
		(I, tree) = commons.inspect(('node_ids', 'knn_tree'))(self.knn)
		return [I[j] for j in tree.query_radius([self.stop_pos[stopid]], r=PARAM['walker_max_busstop_distance'])[0] if (stopid != I[j])]

	def where_can_i_go(self, A, t0, tspan=None) :
		return {
			B : {
				'A' : A,
				'a' : self.stop_pos[A],
				'B' : B,
				'b' : self.stop_pos[B],
				't0' : t0,
				't1' : t0 + dt.timedelta(seconds=(5 + commons.geodesic(self.stop_pos[A], self.stop_pos[B]) / PARAM['walker_speed'])),
				'transit' : {'mode': "walk"},
			}
			for B in self.get_neighbors(A)
		}

	def __del__(self) :
		pass


class BusstopBusser :
	def __init__(self, routes) :
		self.routes = { ROUTE_KEY(route) : route for route in routes }
		self.routes_from = defaultdict(set)
		for (i, route) in self.routes.items() :
			assert(type(route['Direction']) is int), "Routes should be unidirectional at this point"
			# Note: we are excluding the last stop (no boarding there)
			for stop in route['Stops'][:-1] :
				self.routes_from[stop['StopUID']].add(i)
		self.routes_from = dict(self.routes_from)
		#print(list(self.routes_from.items())[0:10])

	def where_can_i_go(self, A, t0, tspan=dt.timedelta(minutes=60)) :
		dest = dict()

		if A not in self.routes_from :
			# print("Warning: No routes from bus stop {}".format(A))
			return dest
		else :
			routes = self.routes_from[A]

		# t0 = np.datetime64(t0)
		# tspan = np.timedelta64(tspan)

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
			tt = tt.loc[(t0 <= tt[A]) & (tt[A] <= (t0 + tspan)), A:]

			if tt.empty :
				continue

			# This should not happen, but just in case
			if (len(tt.columns) < 2) :
				continue

			B = tt.columns[1]

			fastest: pd.Series
			# Note: Convert to numpy datetime for the argmin computation
			fastest = tt.ix[tt[B].astype(np.datetime64).idxmin()]

			transit_info = {
				't0' : fastest[A].to_pydatetime(),
				't1' : fastest[B].to_pydatetime(),
				'a' : None, # Stop location unknown here
				'b' : None, # Stop location unknown here
				'A' : A,
				'B' : B,
				'transit' : {'mode': "bus", 'route': route_key, 'bus_id': fastest.name},
			}

			if B in dest :
				dest[B] = min(dest[B], transit_info, key=commons.inspect('t1'))
			else :
				dest[B] = transit_info

		return dest

## ==================== TESTS :

def test1() :
	t0 = PARAM['TZ'].localize(dt.datetime(year=2018, month=11, day=6, hour=13, minute=15))
	print(t0)

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


	bb = BusstopBusser(routes)

	# print(bb.where_can_i_go('KHH308', t))
	# exit(39)

	# print("Routes passing through {}: {}".format(A, bb.routes_from[A]))
	# print(bb.routes[('KHH100', 0)]['Stops'])
	# exit(39)

	bw = BusstopWalker(stop_pos)
	# print(bw.where_can_i_go('KHH308', t))
	# bw.where_can_i_go('KHH380', t)


	(A, Z) = commons.random_subset(stop_pos.keys(), k=2)

	# Initial openset -- start locations
	AA = { A } # 'KHH4560', 'KHH308', 'KHH12822'
	# Target locations
	ZZ = { Z } # 'KHH4391'

	# # Long search, retakes same busroute
	# (AA, ZZ) = ({'KHH3820'}, {'KHH4484'})

	# Relatively short route, four buses
	(AA, ZZ) = ({'KHH4439'}, {'KHH4370'})

	print("Finding a route from {} to {}".format(AA, ZZ))



	import matplotlib.pyplot as plt
	plt.ion()
	(fig, ax) = plt.subplots()


	g = nx.DiGraph()

	# A*-algorithm heuristic: cost estimate from C to Z
	# It is "admissible" if it never over-estimates
	def h(C) :
		return min(
			dt.timedelta(seconds=(commons.geodesic(stop_pos[C], stop_pos[Z]) / (3 * PARAM['walker_speed'])))
			for Z in ZZ
		)

	# A*-algorithm cost estimator of path via C
	def f(C) :
		return g.nodes[C]['t'] + h(C)

	openset = set(AA)

	for P in openset :
		g.add_node(P, t=t0.astimezone(dt.timezone.utc).replace(tzinfo=None), pos=ll2xy(stop_pos[P]))

	# Next figure update
	nfu = dt.datetime.now()

	while openset :

		# A*-algorithm: select candidate node/path to extend
		C = min(openset, key=f)

		openset.remove(C)


		ti_choices = chain.from_iterable(
			mode.where_can_i_go(C, t0=g.nodes[C]['t']).values()
			for mode in [bb, bw]
		)

		for ti in ti_choices :

			B = ti['B']

			if B in g.nodes :
				if (g.nodes[B]['t'] <= ti['t1']) :
					continue
				else :
					g.remove_edges_from(list(g.in_edges(B)))

			g.add_node(B, t=ti['t1'], pos=ll2xy(stop_pos[B]), ti=deepcopy(ti))
			g.add_edge(C, B, mode=ti['transit']['mode'])
			openset.add(B)

			#print("Added new path to:", B)

			if B in ZZ :
				#print("Done!")
				openset.clear()

				itinerary = []
				while (B not in AA) :
					itinerary.append(deepcopy(g.nodes[B]['ti']))
					B = itinerary[-1]['A']
				itinerary.reverse()

				for it in itinerary :
					t: dt.datetime = pytz.utc.localize(it['t0'])
					t = t.astimezone(tz=PARAM['TZ'])
					print("{} : {} {}".format(t.strftime("%Y%m%d %H:%M (%Z)"), it['transit']['mode'], it['transit']))

				break

		if (dt.datetime.now() >= nfu) or (not openset) :
			ax: plt.Axes
			ax.cla()

			nx.draw_networkx_edges(g, ax=ax, edgelist=[(a, b) for (a, b, d) in g.edges.data('mode') if (d == "walk")], pos=nx.get_node_attributes(g, 'pos'), edge_color='g', arrowsize=5, node_size=0)
			nx.draw_networkx_edges(g, ax=ax, edgelist=[(a, b) for (a, b, d) in g.edges.data('mode') if (d == "bus" )], pos=nx.get_node_attributes(g, 'pos'), edge_color='b', arrowsize=5, node_size=0)
			if AA :
				ax.plot(*zip(*[ll2xy(stop_pos[P]) for P in AA]), 'go')
			if ZZ :
				ax.plot(*zip(*[ll2xy(stop_pos[P]) for P in ZZ]), 'ro')
			if openset :
				ax.plot(*zip(*[ll2xy(stop_pos[O]) for O in openset]), 'kx')

			if not openset :
				try :
					for it in itinerary :
						(y, x) = zip(stop_pos[it['A']], stop_pos[it['B']])
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

