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

	'MOTC_routes' : "OUTPUT/00/ORIGINAL_MOTC/{city}/CityBusApi_StopOfRoute.json",
	'MOTC_shapes' : "OUTPUT/00/ORIGINAL_MOTC/{city}/CityBusApi_Shape.json",
	'MOTC_stops'  : "OUTPUT/00/ORIGINAL_MOTC/{city}/CityBusApi_Stop.json",

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
				't1' : t0 + dt.timedelta(seconds=commons.geodesic(self.stop_pos[A], self.stop_pos[B]) / PARAM['walker_speed']),
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
			print("Warning: No routes from bus stop {}".format(A))
			return dest
		else :
			routes = self.routes_from[A]

		t0 = np.datetime64(t0)
		tspan = np.timedelta64(tspan)
		for route_key in routes :

			fn = IFILE['timetable_json'].format(**dict(zip(['routeid', 'dir'], route_key)))
			if not os.path.isfile(fn) :
				print("Warning: No timetable for route {}".format(route_key))
				continue
			# Note: 'datetime64[ms]' appears necessary here to properly parse the JSON
			#       but the resulting datatype is '...[ns]'
			tt : pd.DataFrame # Timetable
			tt = pd.read_json(commons.zipjson_load(fn)['timetable_df'], dtype='datetime64[ms]')
			tt = tt.loc[(t0 <= tt[A]) & (tt[A] <= (t0 + tspan)), A :]

			if tt.empty :
				continue

			#tt = tt.sort_values(by=A)
			assert(len(tt.columns) >= 2)

			B = tt.columns[1]

			fastest = tt.ix[tt[B].idxmin()]
			transit_info = {
				't0' : fastest[A].to_pydatetime(), #, tzinfo=dt.timezone.utc),
				't1' : fastest[B].to_pydatetime(), #, tzinfo=dt.timezone.utc),
				'a' : None, # Stop location unknown here
				'b' : None, # Stop location unknown here
				'A' : A,
				'B' : B,
				'transit' : {'mode': "bus", 'id': route_key},
			}
			if B in dest :
				dest[B] = min(dest[B], transit_info, key=commons.inspect('t1'))
			else :
				dest[B] = transit_info

		return dest

## ==================== TESTS :

def test1() :
	t0 = dt.datetime(year=2018, month=11, day=6, hour=13, minute=15, tzinfo=PARAM['TZ'])
	print(t0)

	# Initial openset -- start locations
	AA = { 'KHH4560' } #'KHH308', 'KHH12822'
	# Target locations
	ZZ = { 'KHH4391' }

	#
	t0 = t0.astimezone(dt.timezone.utc).replace(tzinfo=None)

	def ll2xy(latlon) :
		return (latlon[1], latlon[0])

	routes = commons.zipjson_load(IFILE['MOTC_routes'])
	bb = BusstopBusser(routes)

	# print(bb.where_can_i_go('KHH308', t))
	# exit(39)

	# print("Routes passing through {}: {}".format(A, bb.routes_from[A]))
	# print(bb.routes[('KHH100', 0)]['Stops'])
	# exit(39)


	stops = commons.zipjson_load(IFILE['MOTC_stops'])

	stop_pos = {
		stop['StopUID'] : commons.inspect({'StopPosition' : ('PositionLat', 'PositionLon')})(stop)
		for stop in stops
	}

	bw = BusstopWalker(stop_pos)
	# print(bw.where_can_i_go('KHH308', t))
	# bw.where_can_i_go('KHH380', t)


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
		g.add_node(P, t=t0, pos=ll2xy(stop_pos[P]))

	# Next figure update
	nfu = dt.datetime.now()

	while openset :

		# A*-algorithm: select candidate node/path to extend
		C = min(openset, key=f)

		openset.remove(C)


		ti_next = chain.from_iterable(
			mode.where_can_i_go(C, t0=g.nodes[C]['t']).values()
			for mode in [bb, bw]
		)

		for ti in ti_next :

			B = ti['B']

			if B in g.nodes :
				if (g.nodes[B]['t'] <= ti['t1']) :
					continue
				else :
					g.remove_edges_from(list(g.in_edges(B)))

			g.add_node(B, t=ti['t1'], pos=ll2xy(stop_pos[B]))
			g.add_edge(C, B, mode=ti['transit']['mode'])
			openset.add(B)

			#print("Added new path to:", B)

			if B in ZZ :
				#print("Done!")
				openset = {}
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

			plt.pause(0.1)

			nfu = dt.datetime.now() + dt.timedelta(seconds=2)

	plt.ioff()
	plt.show()



## ===================== WORK :


## ==================== ENTRY :

if (__name__ == "__main__") :
	test1()

