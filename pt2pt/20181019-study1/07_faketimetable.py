#!/usr/bin/python3

# RA, 2018-10-21

## ================== IMPORTS :

import os
import pickle
import inspect
import numpy as np
import networkx as nx
import pint
import pandas as pd

## ==================== NOTES :

pass

## ==================== INPUT :

IFILE = {
	'OSM-pickled': "OUTPUT/02/UV/kaohsiung.pkl",
}

## =================== OUTPUT :

OFILE = {
	'timetable': "OUTPUT/07/timetable_{route}.txt",
}

# Create output directories
for f in OFILE.values() : os.makedirs(os.path.dirname(f), exist_ok=True)


## ==================== PARAM :

PARAM = {
	'': 0,
}

## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))


# Log which files are opened
def logged_open(filename, mode='r', *argv, **kwargs):
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

	def s(i, j) :
		if (i < 0) or (j < 0) : return 0
		return S[i, j]

	# Dynamic programing loops
	for i in range(0, M.shape[0]) :
		for j in range(0, M.shape[1]) :
			(S[i, j], I[i, j], J[i, j]) = \
				(
					# In the first column, can only go up
					(j == 0) and (s(i - 1, j) + M[i, j], i - 1, j)
				) or (
					# Otherwise have a choice:
					min(
						# go left
						(s(i, j - 1), i, j - 1),
						# go up
						(s(i - 1, j) + M[i, j], i - 1, j)
					)
				)

	# Retrace the optimal way
	match = [None] * M.shape[0]
	while (i >= 0) :
		M[i, j] = max(M.flatten()) # For visualization below
		match[i] = j
		(i, j) = (I[i, j], J[i, j])

	# # For visualization:
	# import matplotlib.pyplot as plt
	# plt.imshow(M)
	# plt.show()

	# Now: row i is matched with column match[i]
	return match


## ===================== WORK :

def concat_route(osm, route):
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
	pieces = [way_nodes[wid] for wid in route_ways]

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
	assert (type(compoway) is list)

	if not compoway:
		raise RuntimeError("Possibly orientation problem in the first piece of route {}.".format(route_name))

	# Attach the remaining pieces
	for piece in pieces[1:]:

		# Last visited node
		z = compoway[-1]

		# Can it connect to the next piece?
		if not (z in piece[True]):
			raise RuntimeError("Cannot connect to the next piece (route {}).".format(route_name))

		# Choose the orientation that connects
		way = piece[z == piece[True][0]]
		assert (type(way) is list)

		if not way:
			raise RuntimeError("Apparent orientation mismatch of route {} at {}".format(route_name,
			                                                                            [way_tags[wid] for wid in
			                                                                             route_ways if
			                                                                             (way_nodes[wid] == piece)]))

		if not (compoway[-1] == way[0]):
			raise RuntimeError("In route {}, cannot connect piece {}".format(route_name,
			                                                                 [way_tags[wid] for wid in route_ways if
			                                                                  (way_nodes[wid] == piece)]))

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


def fake_timetable():
	# Load road graph, osm relations, etc.
	filename = IFILE['OSM-pickled']
	osm = pickle.load(logged_open(filename, 'rb'))

	# Extract useful data
	G = osm['G']
	node_tags = osm['node_tags']
	locs = osm['locs']
	rels = osm['rels']
	way_tags = osm['way_tags']
	way_nodes = osm['way_nodes']

	# Nodes of platforms (near road),
	# stop nodes (on the road),
	# and the composite route way
	# indexed by route ID
	route_geo = {  }

	print("Making time tables...")

	for (route_id, route) in rels['route'].items():
		# Skip non-bus routes
		if not (route['t'].get('route') == 'bus'): continue

		route_name = route['t'].get('name')
		assert (route_name is not None)

		# # For debugging purposes
		# if not (route_name == "環狀東線") : continue

		try:

			routeway = concat_route(osm, route)

		except RuntimeError as e:

			print(e)
			continue

		route_platforms = route['n']

		print("Got route {} with {} stops.".format(route_name, len(route_platforms)))

		# Match bus stop to a location on the route
		from scipy.spatial.distance import cdist as dist_matrix
		from geopy.distance import geodesic as geo_dist
		d = dist_matrix(
			[locs[i] for i in route_platforms],
			[locs[j] for j in routeway],
			metric=(lambda p, q: geo_dist(p, q).m)
		)
		# Note: alternatively, could match to route segments
		# Note: the following could be ambiguous due to repeated stops
		# m = { route_platforms[i] : routeway[j] for (i, j) in enumerate(align(d)) }
		m = align(d)
		# print("BusStop-to-RouteNode match:", m)
		# Measure distance only along the route
		g = nx.subgraph(G, routeway)
		# Distance stop0-to-stopN
		stopdist = [0] + list(np.cumsum([
			nx.shortest_path_length(g, source=routeway[a], target=routeway[b], weight='len')
			for (a, b) in zip(m[:-1], m[1:])
		]))

		route_geo[route_id] = {
			# Platforms along the route (OSM node IDs)
			'platform' : route_platforms,
			# Stop names
			'stopname' : [node_tags[n].get('name') for n in route_platforms],
			# Estimated stop nodes on the route way (OSM node IDs)
			'stopnode' : [routeway[j] for j in m],
			# The composite route way (OSM node IDs)
			'routeway' : routeway,
			# Accumulated travel distance for stop nodes in meters
			'stopdist' : stopdist,
		}

		# Visualization
		# A long circular route
		if False and (route_name == "環狀東線") :
			# Visualization of stop distance
			import matplotlib.pyplot as plt
			# Average bus speed in km/h
			v = 17
			# Arrival times in minutes
			stoptime = np.asarray(stopdist) / (v * 1000 / 60)
			# For reference, the first sunday bus (http://timetable.ibus.com.tw/168E.pdf)
			stoptime_ref = [0, 2, 4, 6, 7, 8, 10, 12, 13, 15, 17, 19, 20, 21, 22, 23, 25, 26, 28, 29, 31, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 47, 48, 50, 51, 52, 53, 54, 55, 56, 58, 60, 62, 63, 64, 66, 68, 70, 71, 73, 75, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 100, 103, 105, 105, 105, 106]

			plt.plot(stoptime, '.-')
			plt.plot(stoptime_ref, '.-')
			plt.title(route_name)
			plt.show()


	# Got route_geo

	print("Writing timetables...")

	from datetime import timedelta
	import random
	for (route_id, geodata) in route_geo.items() :
		u = pint.UnitRegistry()
		# Average bus velocity
		v = 17 * (u.km / u.hour)
		# Accumulated travel distance
		d = geodata['stopdist'] * u.meter
		# Check that the array has units of length
		d.to(u.meter)
		# Accumulated travel time
		T0 = (d / v).to(u.minute)
		# Bus frequency
		busfreq = 4 / u.hour
		# First bus
		t = timedelta(hours=6, minutes=random.randint(0, 30))
		# Last possible bus
		tz = timedelta(hours=23, minutes=0)

		# Time table
		TT = pd.DataFrame(columns=range(len(geodata['platform'])))
		TT.name = "Route {}".format(rels['route'][route_id]['t'].get('name'))
		TT = TT.rename_axis("> Stop number >", axis='columns')

		while (t <= tz) :
			# Times of passage for this bus
			T = [pd.Timedelta(seconds=s) for s in (T0 + (t.seconds * u.second)).to(u.second).magnitude]
			TT = TT.append([ T ], ignore_index=True, sort=False)
			# Next bus starts at:
			t = t + timedelta(hours=(1 / busfreq).to(u.hour).magnitude)

		filename = OFILE['timetable'].format(route=route_id)
		with open(filename, 'w') as f :
			TT.to_csv(f, sep='\t', index=False, header=False)

		# Read with
		# df.apply(pd.to_timedelta, pd.read_csv(filename, sep='\t', header=None, index_col=None), axis=0)

## ==================== ENTRY :

if (__name__ == "__main__"):
	fake_timetable()
