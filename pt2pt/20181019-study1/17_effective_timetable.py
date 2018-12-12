#!/usr/bin/python3

# RA, 2018-12-11


## ================== IMPORTS :

from helpers import commons, graph

from math import sqrt, floor

import re
import json
import random
import inspect
import traceback
import networkx as nx
import numpy as np
import datetime as dt
import dateutil.parser
from sklearn.neighbors import NearestNeighbors

from itertools import chain, product, groupby

from difflib import SequenceMatcher
from sklearn.cluster import AgglomerativeClustering

import matplotlib as mpl


## ==================== NOTES :

pass


## =============== DIAGNOSTIC :

open = commons.logged_open


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

	dutystatus = 'DutyStatus'
	busstatus = 'BusStatus'


# Helpers
BUSID_OF = (lambda b : b[KEYS.busid])
ROUTEID_OF = (lambda r : r[KEYS.routeid])
DIRECTION_OF = (lambda r : r[KEYS.dir])

# What finally identifies a one-way route
ROUTE_KEY = (lambda r : (ROUTEID_OF(r), DIRECTION_OF(r)))


## ==================== INPUT :

IFILE = {
	#'mapmatched' : "OUTPUT/14/mapmatched/{scenario}/{routeid}-{direction}/UV/{mapmatch_uuid}.{ext}",

	'segment_by_route' : "OUTPUT/13/{scenario}/byroute/UV/{routeid}-{dir}.json",

	'MOTC_routes'      : "OUTPUT/00/ORIGINAL_MOTC/{City}/CityBusApi_StopOfRoute.json",
}


## =================== OUTPUT :

OFILE = {
}

commons.makedirs(OFILE)


## ==================== PARAM :

PARAM = {
	#'mapbox_api_token' : open(".credentials/UV/mapbox-token.txt", 'r').read(),

}


## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))


## =================== SLAVES :



# At what time does a given bus visit the bus stops?
def bus_at_stops(run, stops) :
	mpl.use('GTK3Agg')
	import matplotlib.pyplot as plt

	# TODO: some candidate_tdt have large time-gaps

	# These are sparse samples of a bus trajectory
	candidate_gps = list(map(tuple, run['BusPosition']))
	# Timestamps of GPS records as datetime objects
	candidate_tdt = list(map(dateutil.parser.parse, run['GPSTime']))

	# These are fixed platform locations
	reference_gps = list(commons.inspect({'StopPosition': ('PositionLat', 'PositionLon')})(stop) for stop in stops)

	# Goal: obtain an estimate ref_guess_tdt of when the bus is nearest to the platforms

	#print(candidate_gps)
	#print(run['GPSTime'])
	#print(candidate_tdt)
	#print(reference_gps)

	segments = list(zip(candidate_gps[:-1], candidate_gps[1:]))
	segm_tdt = list(zip(candidate_tdt[:-1], candidate_tdt[1:]))

	M = np.vstack([graph.dist_to_segment(r, s)[0] for s in segments] for r in reference_gps)

	if not M.size : return

	# M contains distances in meters
	# We expect the bus to travel about 17km/h on average, say 5m/s
	# So, a penalty of p = 0.1m/s should not make much of a difference,
	# unless the bus is idling

	# Penalty rate
	p = 0.1 # m/s
	# Penalty matrix
	P = -np.vstack(len(reference_gps) * [np.cumsum([p * (t1 - t0).seconds for (t0, t1) in segm_tdt])])

	# Add idling penalty
	M += P

	match = commons.align(M)
	print(match)

	segments = [segments[j] for j in match]
	segm_tdt = [segm_tdt[j] for j in match]
	seg_dist = [graph.dist_to_segment(r, s) for (r, s) in zip(reference_gps, segments)]

	ref_guess_tdt = [
		t0 + q * (t1 - t0)
		for ((d, q), (t0, t1)) in zip(seg_dist, segm_tdt)
	]

	#for t in ref_guess_tdt : print(t)

	print("ETA (min):", [round((t - ref_guess_tdt[0]).seconds/60) for t in ref_guess_tdt])

	is_monotone = all((s <= t) for (s, t) in zip(ref_guess_tdt, ref_guess_tdt[1:]))
	#assert(is_monotone), "Time stamps not monotone"
	print("Monotonicity of ETA:", is_monotone)

	(fig, ax_array) = plt.subplots(nrows=1, ncols=2)
	(ax1, ax2) = ax_array

	ax1.imshow(M)

	(y, x) = zip(*candidate_gps)
	ax2.plot(x, y, '-')
	(y, x) = zip(*reference_gps)
	ax2.plot(x, y, 'bo')

	while plt.fignum_exists(fig.number) :
		try :
			plt.pause(0.1)
		except :
			break

	plt.close(fig)


## =================== MASTER :

def generate_timetables() :
	motc_routes = commons.index_dicts_by_key(commons.zipjson_load(IFILE['MOTC_routes'].format(City="Kaohsiung")), ROUTE_KEY)

	run_files = commons.ls(IFILE['segment_by_route'].format(scenario="**", routeid="*", dir="*"))
	print("Found {} route files.".format(len(run_files)))


	for run_file in run_files :
		print("===")
		print("Analyzing route file {}.".format(run_file))

		(scenario, routeid, dir) = re.fullmatch(IFILE['segment_by_route'].format(scenario="(.*)", routeid="(.*)", dir="(.*)"), run_file).groups()
		dir = int(dir)
		print("Route: {}, direction: {} (from scenario: {})".format(routeid, dir, scenario))

		# Load all bus run segments for this case
		runs = commons.zipjson_load(run_file)
		print("Number of runs: {} ({})".format(len(runs), "total"))

		runs = [run for run in runs if (run.get('quality') == "+")]

		try :
			stops = motc_routes[(routeid, dir)]['Stops']
		except KeyError :
			print("Warning: No stops info for route {}, direction {}".format(routeid, dir))
			continue

		for run in runs :
			bus_at_stops(run, stops)

		# import matplotlib.pyplot as plt
		# plt.ioff()
		# plt.show()
		#exit(39)


## ==================== ENTRY :

if (__name__ == "__main__") :
	generate_timetables()
