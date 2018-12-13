#!/usr/bin/python3

# RA, 2018-12-11


## ================== IMPORTS :

from helpers import commons, graph

from math import sqrt, floor, ceil

import re
import json
import random
import inspect
import traceback
import networkx as nx
import numpy as np
import pandas as pd
import datetime as dt
import dateutil.parser
from sklearn.neighbors import NearestNeighbors

from multiprocessing import cpu_count
from joblib import Parallel, delayed

from progressbar import progressbar

import pint

from itertools import chain, product, groupby

from difflib import SequenceMatcher
from sklearn.cluster import AgglomerativeClustering

import matplotlib as mpl


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
	'timetable' : "OUTPUT/17/timetable/{scenario}/{routeid}-{dir}.{ext}",
}

commons.makedirs(OFILE)


## ==================== PARAM :

PARAM = {
	#'mapbox_api_token' : open(".credentials/UV/mapbox-token.txt", 'r').read(),

	# When is the bus run too short at the tails?
	'tail_eta_patch_dist' : 50, # meters
	# Have at least those many waypoints close to the run
	'min_near_run' : 3,

	'n_parallel_jobs' : min(12, ceil(cpu_count() / 1.5)),
}


## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))


## =================== SLAVES :



# At what time does a given bus visit the bus stops?
def bus_at_stops(run, stops) :
	# Note: pint does not serialize well
	Units = pint.UnitRegistry()

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

	assert(M.size), "Undefined behavior for missing data"

	# M contains distances in meters
	# We expect the bus to travel about 17km/h on average, say 5m/s
	# So, a penalty of p = 0.1m/s should not make much of a difference,
	# unless the bus is idling

	# Penalty rate
	p = -0.1 # m/s
	# Penalty matrix
	P = np.vstack(len(reference_gps) * [np.cumsum([p * (t1 - t0).seconds for (t0, t1) in segm_tdt])])

	# Add idling penalty
	M += P

	# Make a copy for imshow
	M_image = M.copy()

	match = commons.align(M_image)

	segments = [segments[j] for j in match]
	segm_tdt = [segm_tdt[j] for j in match]
	seg_dist = [graph.dist_to_segment(r, s) for (r, s) in zip(reference_gps, segments)]

	ref_guess_tdt = [
		t0 + q * (t1 - t0)
		for ((d, q), (t0, t1)) in zip(seg_dist, segm_tdt)
	]

	if not (PARAM['min_near_run'] <= sum((d <= PARAM['tail_eta_patch_dist']) for (d, q) in seg_dist)) :
		return [None] * len(stops)

	try :
		# Patch ETA at the tails
		typical_speed = np.median([s for s in run['Speed'] if s]) * (Units.km / Units.hour)
		# Look at front and back tails
		for (direction, traversal) in [(+1, commons.identity), (-1, reversed)] :
			# Indices and distances
			tail = [(n, d) for (n, (d, q)) in traversal(list(enumerate(seg_dist)))]
			# Tail characterized by large distances
			tail = tail[0:(1 + min(i for (i, (n, d)) in enumerate(tail) if (d <= PARAM['tail_eta_patch_dist'])))]
			# Indices of the tail
			tail = [n for (n, d) in tail]
			if not tail : continue
			# First visited waypoint
			first = tail.pop()
			# Patch ETA for prior waypoints
			for n in tail :
				td = dt.timedelta(seconds=(commons.geodesic(reference_gps[n], reference_gps[first]) * Units.meter / typical_speed).to(Units.second).magnitude)
				ref_guess_tdt[n] = ref_guess_tdt[first] - direction * td
	except :
		print("Warning: Patching tail ETA failed")
		print(traceback.format_exc())

	is_monotone = all((s < t) for (s, t) in zip(ref_guess_tdt, ref_guess_tdt[1:]))
	#assert(is_monotone), "Time stamps not monotone"

	# Diagnostics
	if False :
		mpl.use('GTK3Agg')
		import matplotlib.pyplot as plt

		print("Match:", match)

		print("Dist wpt to closest seg:", seg_dist)

		print("ETA (sec):", [round((t - ref_guess_tdt[0]).total_seconds()) for t in ref_guess_tdt])
		print("Strict monotonicity of ETA:", is_monotone)

		(fig, ax_array) = plt.subplots(nrows=1, ncols=2)
		(ax1, ax2) = ax_array

		ax1.imshow(M_image)

		(y, x) = zip(*candidate_gps)
		ax2.plot(x, y, '-')

		(y, x) = zip(*reference_gps)
		ax2.plot(x, y, 'bo')

		for ((wy, wx), ((ay, ax), (by, bx)), (d, q)) in zip(reference_gps, segments, seg_dist) :
			x = (wx, ax + q * (bx - ax))
			y = (wy, ay + q * (by - ay))
			ax2.plot(x, y, 'r-')

		while plt.fignum_exists(fig.number) :
			try :
				plt.pause(0.1)
			except :
				break

		plt.close(fig)

	# Sanity check
	assert(len(ref_guess_tdt) == len(stops)), "Stop and ETA vector length mismatch"

	return ref_guess_tdt


## =================== MASTER :

def generate_timetables() :
	motc_routes = commons.index_dicts_by_key(commons.zipjson_load(IFILE['MOTC_routes'].format(City="Kaohsiung")), ROUTE_KEY)

	run_files = commons.ls(IFILE['segment_by_route'].format(scenario="**", routeid="*", dir="*"))
	print("Found {} route files.".format(len(run_files)))


	for run_file in run_files :
		print("===")
		print("Analyzing route file {}.".format(run_file))

		(scenario, routeid, dir) = re.fullmatch(IFILE['segment_by_route'].format(scenario="(.*)", routeid="(.*)", dir="(.*)"), run_file).groups()
		case = {'scenario': scenario, 'routeid': routeid, 'dir': int(dir)}
		print("Route: {routeid}, direction: {dir} (from scenario: {scenario})".format(**case))

		# Load all bus run segments for this case
		runs = commons.zipjson_load(run_file)
		print("Number of runs: {} ({})".format(len(runs), "total"))

		runs = [run for run in runs if (run.get('quality') == "+")]

		print("Number of runs: {} ({})".format(len(runs), "quality"))

		try :
			route = motc_routes[(case['routeid'], case['dir'])]
			stops = route['Stops']
		except KeyError :
			print("Warning: No stops info for route {routeid}, direction {dir}".format(**case))
			continue

		# ETA table of Busrun x Stop
		ETA = np.vstack(Parallel(n_jobs=PARAM['n_parallel_jobs'])(delayed(bus_at_stops)(run, stops) for run in progressbar(runs)))

		# pandas does not digest dt.datetime
		# https://github.com/pandas-dev/pandas/issues/13287
		ETA = ETA.astype(np.datetime64)

		# Timetable as DataFrame
		df = pd.DataFrame(data=ETA, columns=[s['StopUID'] for s in stops])

		J = {
			**case,
			'route' : route,
			'run_file' : run_file,
			'timetable_df' : df.to_json(),
		}

		with open(commons.makedirs(OFILE['timetable'].format(ext="json", **case)), 'w') as fd :
			json.dump(J, fd)


## ==================== ENTRY :

if (__name__ == "__main__") :
	generate_timetables()
