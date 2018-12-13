#!/usr/bin/python3

# RA, 2018-10-21

## ================== IMPORTS :

from helpers import commons, graph, maps

import pickle
import inspect
import numpy as np
import networkx as nx

import sklearn.neighbors

from joblib import Parallel, delayed
from progressbar import progressbar
from itertools import chain

from collections import defaultdict

## ==================== NOTES :

pass


## ===================== META :

# What finally identifies a one-way route
ROUTE_KEY = (lambda r : (r['SubRouteUID'], r['Direction']))


## ================== PARAM 1 :

PARAM = {
	'City' : "Kaohsiung",
	'scenario' : "Kaohsiung/20181105-20181111",
}


## ==================== INPUT :

IFILE = {
	#'OSM-pickled' : "OUTPUT/02/UV/kaohsiung.pkl",

	'MOTC_routes' : "OUTPUT/00/ORIGINAL_MOTC/{City}/CityBusApi_StopOfRoute.json",
	'MOTC_shapes' : "OUTPUT/00/ORIGINAL_MOTC/{City}/CityBusApi_Shape.json",
	'MOTC_stops'  : "OUTPUT/00/ORIGINAL_MOTC/{City}/CityBusApi_Stop.json",

	'timetable_json' : "OUTPUT/17/timetable/{scenario}/json/{{routeid}}-{{dir}}.json",
}

for (k, s) in IFILE.items() : IFILE[k] = s.format(**PARAM)


## =================== OUTPUT :

OFILE = {
	'' : "",
}


## ================== PARAM 2 :

PARAM.update({
	'walkable_busstop_distance' : 250, # meters
})


## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))


## =================== SLAVES :

class BusstopWalker :

	def __init__(self, stops, loops=True) :
		stops = {
			stop['StopUID'] : commons.inspect({'StopPosition' : ('PositionLat', 'PositionLon')})(stop)
			for stop in stops
		}

		self.g = nx.DiGraph()

		(I, tree) = commons.inspect(('node_ids', 'knn_tree'))(graph.compute_geo_knn(stops))
		tree : sklearn.neighbors.BallTree

		# BusStop with ID 'i' and geo-location 'p'
		def walkable_edges(i, p) :
			return [(i, I[j]) for j in tree.query_radius([p], r=PARAM['walkable_busstop_distance'])[0] if (loops or (i != I[j]))]

		self.g.add_edges_from(chain.from_iterable(
			walkable_edges(i, p) for (i, p) in progressbar(stops.items())
			#Parallel(n_jobs=2, batch_size=100)(delayed(walkable_edges)(i, p) for (i, p) in progressbar(stops.items()))
		))

		nx.set_node_attributes(self.g, stops, name='pos')
		nx.set_edge_attributes(self.g, {(a, b): commons.geodesic(stops[a], stops[b]) for (a, b) in self.g.edges}, name='len')

	def debug_show_graph_exerpt(self) :
		print(list(self.g.nodes.data('pos'))[0:10])
		print(list(self.g.edges.data('len'))[0:10])

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
		print(list(self.routes_from.items())[0:10])

	def where_can_i_go(self, stopid, t):
		routes = self.routes_from(stopid)
		pass

## ==================== TESTS :

def test1() :
	routes = commons.zipjson_load(IFILE['MOTC_routes'])
	bb = BusstopBusser(routes)
	exit(39)

	stops = commons.zipjson_load(IFILE['MOTC_stops'])
	bw = BusstopWalker(stops, loops=False)
	bw.debug_show_graph_exerpt()


## ===================== WORK :


## ==================== ENTRY :

if (__name__ == "__main__") :
	test1()

