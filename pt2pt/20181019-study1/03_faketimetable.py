#!/usr/bin/python3

# RA, 2018-10-21

## ================== IMPORTS :

import pickle
import inspect
import networkx as nx

## ==================== NOTES :

pass


## ==================== INPUT :

IFILE = {
	'OSM-pickled' : "OUTPUT/02/UV/kaohsiung.pkl",
}


## =================== OUTPUT :

OFILE = {
	'timetable' : "OUTPUT/03/{route_id}.txt",
}


## ==================== PARAM :

PARAM = {
	'' : 0,
}

## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))

# Log which files are opened
def logged_open(filename, mode='r', *argv, **kwargs) :
	print("({}):\t{}".format(mode, filename))
	return open(filename, mode, *argv, **kwargs)

# Find a way through M bottom-to-top with right-to-left drift
# that minimizes the sum of entries (using dynamic programming)
#
# Recursion template:
#
# def sum(i, j) :
# 	if (i < 0) or (j < 0) : return 0
# 	return min(sum(i, j - 1), M[i, j] + sum(i - 1, j))
#
def align(M) :

	# Sum matrix
	S = 0 * M

	# These will record the trajectory
	import numpy as np
	I = np.zeros(M.shape, dtype=int)
	J = np.zeros(M.shape, dtype=int)

	# Dynamic programing loops
	for i in range(0, M.shape[0]) :
		for j in range(0, M.shape[1]) :
			(S[i, j], I[i, j], J[i, j]) = min(
				((j == 0) and (M[i, j], -1, j)) or (          S[i, j - 1], i, j - 1),
				((i == 0) and (M[i, j], -1, j)) or (M[i, j] + S[i - 1, j], i - 1, j)
			)

	# Retrace the optimal way
	match = [None] * M.shape[0]
	while (i >= 0) :
		match[i] = j
		(i, j) = (I[i, j], J[i, j])

	# Now: row i is matched with column match[i]
	return match

## ===================== WORK :

def concat_route(osm, route) :

	# Extract useful parts from the osm bundle
	G = osm['G']
	way_tags = osm['way_tags']
	way_nodes = osm['way_nodes']

	# Human-readable name for the route
	route_name = route['t'].get('name')

	# Route stops should be graph vertex IDs
	route_stops = route['n']

	# Does the route have at least two stops?
	if (len(route_stops) < 2):
		raise RuntimeError("Route {} has fewer than two stops.".format(route_name))

	# Are all stops also vertices of the graph?
	if not all(stop in G.nodes() for stop in route_stops):
		raise RuntimeError("Stops of route {} not in the graph.".format(route_name))


	# Route 'ways' are pieces of the route
	# The route need not be the shortest one between stops
	# Need to concatenate the 'ways' to get the route
	# Note: typically many more 'ways' than stops
	route_ways = route['w']

	# Need to align the direction of each 'way' piece in the route
	# List of forward-possibly-backward pieces
	pieces = [ way_nodes[wid] for wid in route_ways ]

	if (len(pieces) < 2):
		raise RuntimeError("Route {} has fewer than two pieces.".format(route_name))

	# Do neighboring pieces have nodes in common?
	if not all((len(set(A[True]).intersection(set(B[True]))) > 0) for (A, B) in zip(pieces[:-1], pieces[1:])):
		raise RuntimeError("Route {} seems to have a gap.".format(route_name))

	# Using the first two pieces
	(first, second) = (pieces[0], pieces[1])
	# guess the orientation of the first piece
	compoway = first[first[True][-1] in second[True]]

	#
	assert(type(compoway) is list)

	if not compoway :
		raise RuntimeError("Possibly orientation problem in the first piece of route {}.".format(route_name))

	# Attach the remaining pieces
	for piece in pieces[1:] :

		# Last visited node
		z = compoway[-1]

		# Can it connect to the next piece?
		if not (z in piece[True]) :
			raise RuntimeError("Cannot connect to the next piece (route {}).".format(route_name))

		# Choose the orientation that connects
		way = piece[z == piece[True][0]]
		assert(type(way) is list)

		if not way :
			raise RuntimeError("Apparent orientation mismatch of route {} at {}".format(route_name, [way_tags[wid] for wid in route_ways if (way_nodes[wid] == piece)]))

		if not (compoway[-1] == way[0]) :
			raise RuntimeError("In route {}, cannot connect piece {}".format(route_name, [way_tags[wid] for wid in route_ways if (way_nodes[wid] == piece)]))

		compoway = compoway + way[1:]

	return compoway

	# except Exception as e:
	#
	# 	raise
	#
	# 	# The route could not be patched together from its 'way's
	# 	# Use shortest paths between the stops
	#
	# 	compoway = [route_stops[0]]
	#
	# 	# The following is too naive
	# 	for (a, b) in zip(route_stops[:-1], route_stops[1:]):
	#
	# 		try:
	# 			# Shortest distance between nodes s and t
	# 			p = nx.shortest_path(G, source=a, target=b, weight='len')
	# 		except Exception as e:
	# 			print("Failed to connect stops {} and {}".format(a, b))
	# 			exit()
	#
	# 		# Append piece
	# 		compoway = compoway + p[1:]


def fake_timetable() :
	# Load road graph, osm relations, etc.
	filename = IFILE['OSM-pickled']
	osm = pickle.load(logged_open(filename, 'rb'))

	# Extract useful data
	G = osm['G']
	locs = osm['locs']
	rels = osm['rels']
	way_tags = osm['way_tags']
	way_nodes = osm['way_nodes']


	print("Making time tables")
	for (route_id, route) in rels['route'].items() :
		# Skip non-bus routes
		if not (route['t'].get('route') == 'bus') : continue

		route_name = route['t'].get('name')
		assert(route_name is not None)

		try :

			route_way = concat_route(osm, route)

		except RuntimeError as e :

			print(e)
			continue

		print("Got route", route_name, "with nodes", route_way)

		route_stops = route['n']

		# Match bus stop to a location on the route
		from scipy.spatial.distance import cdist as dist_matrix
		from geopy.distance import geodesic as geo_dist
		d = dist_matrix([locs[i] for i in route_stops], [locs[j] for j in route_way], metric=(lambda p, q : geo_dist(p, q).m))
		# Note: the following could be ambiguous due to repeated stops
		# m = { route_stops[i] : route_way[j] for (i, j) in enumerate(align(d)) }
		m = align(d)
		#print("BusStop-to-RouteNode match:", m)
		print(m)
		for (a, b) in zip(m[:-1], m[1:]) :
			p = nx.shortest_path_length(G, source=route_way[a], target=route_way[b], weight='len')
			print(p)

		exit()


## ==================== ENTRY :

if (__name__ == "__main__") :

	fake_timetable()
