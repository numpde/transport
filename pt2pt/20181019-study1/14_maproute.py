#!/usr/bin/python3

# RA, 2018-11-15

## ================== IMPORTS :

from helpers import commons, maps, graph

from enum import Enum

import uuid
import time
import json
import networkx as nx
import glob
import math
import pickle
import random
import inspect
import datetime as dt
import numpy as np
import dateutil.parser
from itertools import chain

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

## ==================== NOTES :

pass


## ==================== INPUT :

IFILE = {
	'busses' : "OUTPUT/13/Kaohsiung/UV/{busid}.json",
}


## =================== OUTPUT :

OFILE = {
	'progress_img': "OUTPUT/14/UV/progress/current_route.png",
	'progress_txt': "OUTPUT/14/UV/progress/current_route.txt",

	'mapmatched' : "OUTPUT/14/UV/mapmatched/{routeid}/{direction}/{estpathid}.json",
}

commons.makedirs(OFILE)

## ================= METADATA :

# Keys in a realtime JSON record
class KEYS :
	runid = 'RunUUID'

	busid = 'PlateNumb'
	routeid = 'SubRouteUID'
	dir = 'Direction'

	speed = 'Speed'
	azimuth = 'Azimuth'

	time = 'GPSTime'
	pos = 'BusPosition'

	#'bus_stat' : 'BusStatus', # Not all records have this
	#'duty_stat' : 'DutyStatus',

# The subkeys of KEYS.pos
class KEYS_POS :
	lat = 'PositionLat'
	lon = 'PositionLon'

# Helper to extract the Physical-Bus ID
BUSID_OF = (lambda b: b[KEYS.busid])


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
	routeid_of = (lambda r: r[KEYS.routeid])
	direction_of = (lambda r: r[KEYS.dir])

	# Road network (main graph component) with nearest-neighbor tree for the nodes
	g : nx.DiGraph
	(g, knn) = commons.inspect(('g', 'knn'))(
		pickle.load(open(PARAM['OSM_graph_file'], 'rb'))['main_component_with_knn']
	)

	kne = (lambda q : graph.estimate_kne(g, knn, q, ke=20))


	# List of filenames, one file per physical bus, identified by plate number
	bus_files = sorted(glob.glob(IFILE['busses'].format(busid="*")))

	# Filter bus runs by the route ID
	runs = list(
		run
		for fn in bus_files
		for run in commons.zipjson_load(fn)
		if ((routeid_of(run), direction_of(run)) == (route_id, direction))
	)

	print("Found {} runs for route ID {}".format(len(runs), route_id))

	def mm_callback(result) :
		fig: plt.Figure
		ax: plt.Axes

		if not ('plt' in result) :
			(fig, ax) = plt.subplots()
			result['plt'] = { 'fig' : fig, 'ax' : ax }

		if (dt.datetime.now() < result.get('nfu', dt.datetime.min)) :
			return

		(fig, ax) = commons.inspect({'plt' : ('fig', 'ax')})(result)

		ax.cla()

		if ('waypoints' in result) :
			(y, x) = zip(*result['waypoints'])
			ax.plot(x, y, 'o', c='m', markersize=4)

		if ('geo_path' in result) :
			(y, x) = zip(*result['geo_path'])
			ax.plot(x, y, 'b--', linewidth=2, zorder=-50)

		ax.set_title("{} ({}%)".format(result['status'], math.floor(100 * result.get('progress', 0))))

		try :
			fig.savefig(OFILE['progress_img'], bbox_inches='tight', pad_inches=0)
		except Exception as e :
			print("Could not save figure {} ({})".format(OFILE['progress_img'], e))

		# Next figure update
		result['nfu'] = dt.datetime.now() + dt.timedelta(seconds=2)

		if (result['status'] == "done") :
			plt.close(fig)


	# Keep a certain distance between waypoints
	def sparsify(wps, dist=60) :
		a = next(iter(wps))
		yield a
		for b in wps :
			if (graph.geodist(a, b) >= dist) :
				a = b
				yield a

	for run in runs :

		waypoints = list(sparsify(zip(run[KEYS_POS.lat], run[KEYS_POS.lon])))
		if (len(waypoints) < 5) : continue

		mapmatch_attempt = {
			k: run[k] for k in [KEYS.routeid, KEYS.dir, KEYS.runid, KEYS.busid]
		}

		mapmatch_attempt['EstPathUUID'] = uuid.uuid4().hex

		#waypoints = [(22.622249, 120.368713), (22.622039, 120.368301), (22.621929, 120.367332), (22.622669, 120.367736), (22.623569, 120.366722), (22.624959, 120.364402), (22.625329, 120.36338), (22.625549, 120.363357), (22.625379, 120.362777), (22.62565, 120.361061), (22.62594, 120.359947), (22.62602, 120.354911), (22.62577, 120.351226), (22.625219, 120.34732), (22.62494, 120.3442), (22.624849, 120.34317), (22.62597, 120.342582), (22.626169, 120.344428), (22.62811, 120.344451), (22.62968, 120.33908), (22.63017, 120.337562), (22.630279, 120.33715), (22.63042, 120.336341), (22.631919, 120.331932), (22.632989, 120.327766), (22.632789, 120.325233), (22.632829, 120.324371), (22.633199, 120.32283), (22.633449, 120.321639), (22.63459, 120.31707), (22.636629, 120.314437), (22.63758, 120.308952), (22.6375, 120.307777), (22.637899, 120.301162), (22.63788, 120.298866), (22.637899, 120.297393), (22.63718, 120.294151), (22.636989, 120.293609), (22.6354, 120.288566), (22.635179, 120.287719), (22.634139, 120.284576), (22.632179, 120.28379), (22.631229, 120.283309), (22.628789, 120.28199), (22.62845, 120.281806), (22.62507, 120.28054), (22.624259, 120.282028), (22.622869, 120.284973), (22.62247, 120.285827), (22.623029, 120.286407), (22.62531, 120.28524)]
		#waypoints = [(22.62269, 120.367767), (22.623899, 120.366409), (22.626039, 120.359397), (22.62615, 120.357887), (22.62602, 120.35337), (22.625059, 120.345809), (22.625989, 120.342529), (22.625999, 120.343856), (22.626169, 120.344413), (22.628049, 120.344436), (22.628969, 120.340843), (22.62993, 120.338348), (22.63025, 120.337356), (22.63043, 120.337013), (22.631309, 120.334068), (22.63269, 120.329841), (22.63307, 120.328491), (22.63297, 120.326713), (22.632949, 120.324851), (22.63385, 120.319831), (22.637609, 120.307678), (22.637609, 120.305633), (22.63762, 120.304847), (22.637859, 120.300231), (22.63796, 120.297439), (22.63787, 120.296707), (22.63739, 120.294357), (22.637079, 120.293472), (22.6359, 120.289939), (22.63537, 120.288353), (22.634149, 120.284728), (22.629299, 120.28228), (22.62652, 120.280738), (22.62354, 120.283637), (22.622549, 120.28572), (22.622999, 120.28627), (22.625379, 120.285156)])
		#waypoints = [(22.62203, 120.368293), (22.62195, 120.367401), (22.624559, 120.36515), (22.624929, 120.364448), (22.62585, 120.363113), (22.625549, 120.36177), (22.6261, 120.357627), (22.625509, 120.349677), (22.62503, 120.345596), (22.62589, 120.342307), (22.627979, 120.344459), (22.628539, 120.34201), (22.629989, 120.33805), (22.63025, 120.337219), (22.63211, 120.331581), (22.633039, 120.328659), (22.63307, 120.327308), (22.63294, 120.326156), (22.632989, 120.323699), (22.63342, 120.321418), (22.63743, 120.310119), (22.63755, 120.305641), (22.637639, 120.304267), (22.636949, 120.293319), (22.6355, 120.289062), (22.63454, 120.285987), (22.63076, 120.283088), (22.62968, 120.282478), (22.627229, 120.281188), (22.62647, 120.280693), (22.62516, 120.280387), (22.624099, 120.282401), (22.622669, 120.285308), (22.62313, 120.286369), (22.625169, 120.285667)]

		print("waypoints ({}): {}". format(len(waypoints), waypoints))

		commons.seed()
		result = graph.mapmatch(waypoints, g, kne, mm_callback, stubborn=0.2)

		#mapmatch_attempt['mapmatch_result'] = result # Figure not jsonable
		for k in ['geo_path', 'path'] :
			mapmatch_attempt[k] = result[k]

		filename = OFILE['mapmatched'].format(routeid=mapmatch_attempt[KEYS.routeid], direction=mapmatch_attempt[KEYS.dir], estpathid=mapmatch_attempt['EstPathUUID'])
		commons.makedirs(filename)

		try :
			with commons.logged_open(filename, 'w') as fd :
				json.dump(mapmatch_attempt, fd)
		except Exception as e :
			print("Failed to write mapmatch file {} ({})".format(filename, e))

		time.sleep(2)

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
