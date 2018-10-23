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


## ===================== WORK :

def fake_timetable() :
	# Load road graph, osm relations, etc.
	filename = IFILE['OSM-pickled']
	osm = pickle.load(logged_open(filename, 'rb'))

	# Extract useful data
	G = osm['G']
	locs = osm['locs']
	rels = osm['rels']
	way_nodes = osm['way_nodes']

	print("Making time tables")
	for (route_id, route) in rels['route'].items() :
		# Skip non-bus routes
		if not (route['t'].get('route') == 'bus') : continue

		route_name = route['t'].get('name')
		assert(route_name is not None)

		if not (route_name == "覺民幹線(往程)") : continue

		# Route stops are graph vertex ID's
		route_stops = route['n']

		# Does the route have at least two stops?
		if (len(route_stops) < 2) :
			#print(route_name, "has fewer than two stops.")
			continue

		# Are all stops also vertices of the graph?
		if not all(stop in G.nodes() for stop in route_stops) :
			print(route_name, "is not within the graph.")
			continue

		try :

			# Route 'ways' are pieces of the route
			# The route need not be the shortest one between stops
			# Need to concatenate the 'ways' to get the route
			route_ways = route['w']

			way = []

			(a, b) = (route_stops[0], route_stops[-1])

			# Note: typically many more 'ways' than stops

			# Need to align the direction of each 'way' in the "route"
			# List of forward-possibly-backward pieces
			pieces = [ way_nodes[wid] for wid in route['w'] ]

			# We do not have the orientation of the first piece
			(first, second) = (pieces[0], pieces[1])
			compoway = list(first[first[True][-1] in second[True]])

			if not compoway : compoway = list(reversed(first[True])) # CHEAT

			for piece in pieces[1:] :
				# Choose the orientation that connects
				z = compoway[-1]
				assert(z in piece[True])
				way = list(piece[z == piece[True][0]])
				if not way : way = list(reversed(piece[True])) # CHEAT
				assert(compoway[-1] == way[0])
				compoway = compoway + way[1:]
				print(compoway)

		except Exception as e :

			# The route could not be patched together from its 'way's
			# Use shortest paths between the stops

			compoway = [route_stops[0]]

			print(route_name)

			for (a, b) in zip(route_stops[:-1], route_stops[1:]):

				try :
					# Shortest distance between nodes s and t
					p = nx.shortest_path(G, source=a, target=b, weight='len')
				except Exception as e :
					print("Failed to connect stops {} and {}".format(a, b))
					exit()

				# Append piece
				compoway = compoway + p[1:]

			print(route_name, compoway)


## ==================== ENTRY :

if (__name__ == "__main__") :

	fake_timetable()
