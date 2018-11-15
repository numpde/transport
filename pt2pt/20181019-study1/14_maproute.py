#!/usr/bin/python3

# RA, 2018-11-15

## ================== IMPORTS :

from helpers import commons, maps, graph

import time
import json
import networkx as nx
import glob
import pickle
import random
import inspect
import datetime as dt
import dateutil.parser
from itertools import chain


## ==================== NOTES :

pass


## ==================== INPUT :

IFILE = {
	'busses' : "OUTPUT/13/Kaohsiung/UV/{busid}.json",
}


## =================== OUTPUT :

OFILE = {
}

#commons.makedirs(OFILE)

## ================= METADATA :

# Keys in a response file JSON record
KEYS = {
	'busid': 'PlateNumb',

	'routeid': 'SubRouteUID',
	'dir': 'Direction',

	'speed': 'Speed',
	'azimuth': 'Azimuth',

	'time': 'GPSTime',
	'pos': 'BusPosition',

	#'bus_stat' : 'BusStatus', # Not all records have this
	#'duty_stat' : 'DutyStatus',
}

# The subkeys of KEYS['pos']
KEYS_POS = {
	'Lat': 'PositionLat',
	'Lon': 'PositionLon',
}

# Helper to extract the Physical-Bus ID
BUSID_OF = (lambda b: b[KEYS['busid']])


## ==================== PARAM :

PARAM = {
	'mapbox_api_token' : open(".credentials/UV/mapbox-token.txt", 'r').read(),

	'OSM_graph_file' : "OUTPUT/02/UV/kaohsiung.pkl",
}


## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))





## ===================== WORK :

def maproute(route_id, direction) :
	routeid_of = (lambda r: r['SubRouteUID'])
	direction_of = (lambda r: r['Direction'])

	# Road network (main graph component) with nearest-neighbor tree for the nodes
	g : nx.DiGraph
	(g, knn) = commons.inspect(('g', 'knn'))(
		pickle.load(open(PARAM['OSM_graph_file'], 'rb'))['main_component_with_knn']
	)

	kne = (lambda q : graph.estimate_kne(g, knn, q, ke=20))


	# List of filenames, one file per physical bus, identified by plate number
	bus_files = sorted(glob.glob(IFILE['busses'].format(busid="*FT*")))

	# Filter bus runs by the route ID
	runs = list(
		run
		for fn in bus_files
		for run in commons.zipjson_load(fn)
		if (routeid_of(run) == route_id) and (direction_of(run) == direction)
	)

	print("Found {} runs for route ID {}".format(len(runs), route_id))

	import matplotlib.pyplot as plt
	fig: plt.Figure
	ax: plt.Axes

	def mm_callback(result) :

		if (result['status'] == "init") :
			plt.ion()
			plt.show()
			plt.pause(1)

		if (result['status'] == "opti") :
			if (dt.datetime.now() < result.get('nfu', dt.datetime.min)) :
				return

			ax.cla()

			(y, x) = zip(*result['waypoints'])
			ax.plot(x, y, 'o', c='m', markersize=4)

			(y, x) = zip(*result['geo_path'])
			ax.plot(x, y, 'b--', linewidth=2, zorder=-50)

			plt.pause(0.1)

			# Next figure update
			result['nfu'] = dt.datetime.now() + dt.timedelta(seconds=2)

		if (result['status'] == "done") :
			plt.show()
			plt.pause(1)

	# Keep a certain distance between waypoints
	def sparsify(wps) :
		a = next(iter(wps))
		yield a
		for b in wps :
			if (graph.geodist(a, b) >= 50) :
				a = b
				yield a

	for run in runs :
		waypoints = list(sparsify(zip(run['PositionLat'], run['PositionLon'])))
		if (len(waypoints) < 5) : continue

		#waypoints = [(22.622249, 120.368713), (22.622039, 120.368301), (22.621929, 120.367332), (22.622669, 120.367736), (22.623569, 120.366722), (22.624959, 120.364402), (22.625329, 120.36338), (22.625549, 120.363357), (22.625379, 120.362777), (22.62565, 120.361061), (22.62594, 120.359947), (22.62602, 120.354911), (22.62577, 120.351226), (22.625219, 120.34732), (22.62494, 120.3442), (22.624849, 120.34317), (22.62597, 120.342582), (22.626169, 120.344428), (22.62811, 120.344451), (22.62968, 120.33908), (22.63017, 120.337562), (22.630279, 120.33715), (22.63042, 120.336341), (22.631919, 120.331932), (22.632989, 120.327766), (22.632789, 120.325233), (22.632829, 120.324371), (22.633199, 120.32283), (22.633449, 120.321639), (22.63459, 120.31707), (22.636629, 120.314437), (22.63758, 120.308952), (22.6375, 120.307777), (22.637899, 120.301162), (22.63788, 120.298866), (22.637899, 120.297393), (22.63718, 120.294151), (22.636989, 120.293609), (22.6354, 120.288566), (22.635179, 120.287719), (22.634139, 120.284576), (22.632179, 120.28379), (22.631229, 120.283309), (22.628789, 120.28199), (22.62845, 120.281806), (22.62507, 120.28054), (22.624259, 120.282028), (22.622869, 120.284973), (22.62247, 120.285827), (22.623029, 120.286407), (22.62531, 120.28524)]
		#waypoints = [(22.62269, 120.367767), (22.623899, 120.366409), (22.626039, 120.359397), (22.62615, 120.357887), (22.62602, 120.35337), (22.625059, 120.345809), (22.625989, 120.342529), (22.625999, 120.343856), (22.626169, 120.344413), (22.628049, 120.344436), (22.628969, 120.340843), (22.62993, 120.338348), (22.63025, 120.337356), (22.63043, 120.337013), (22.631309, 120.334068), (22.63269, 120.329841), (22.63307, 120.328491), (22.63297, 120.326713), (22.632949, 120.324851), (22.63385, 120.319831), (22.637609, 120.307678), (22.637609, 120.305633), (22.63762, 120.304847), (22.637859, 120.300231), (22.63796, 120.297439), (22.63787, 120.296707), (22.63739, 120.294357), (22.637079, 120.293472), (22.6359, 120.289939), (22.63537, 120.288353), (22.634149, 120.284728), (22.629299, 120.28228), (22.62652, 120.280738), (22.62354, 120.283637), (22.622549, 120.28572), (22.622999, 120.28627), (22.625379, 120.285156)])

		print("waypoints ({}): {}". format(len(waypoints), waypoints))

		(fig, ax) = plt.subplots()

		random.seed(0)
		result = graph.mapmatch(waypoints, g, kne, mm_callback, stubborn=0.2)


def test_map_route() :
	maproute('KHH122', 0)


## ===================== PLAY :

pass


## ================== OPTIONS :

OPTIONS = {
	'MAP_ROUTE' : test_map_route,
}

## ==================== ENTRY :

if (__name__ == "__main__") :
	commons.parse_options(OPTIONS)
