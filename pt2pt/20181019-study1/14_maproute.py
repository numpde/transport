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
from copy import deepcopy
from itertools import chain, product, groupby

import matplotlib as mpl
# Note: do not import pyplot here -- need to select renderer

## ==================== NOTES :

pass


## ==================== INPUT :

IFILE = {
	'busses' : "OUTPUT/13/Kaohsiung/UV/{busid}.json",

	'mapmatched' : "OUTPUT/14/UV/mapmatched/{routeid}/{direction}/{estpathid}",
}


## =================== OUTPUT :

OFILE = {
	'progress_img': "OUTPUT/14/UV/progress/current_route.png",
	'progress_txt': "OUTPUT/14/UV/progress/current_route.txt",

	'mapmatched' : IFILE['mapmatched'],
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

	'map_bbox' : (120.2593, 22.5828, 120.3935, 22.6886),
}


## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))

def run_waypoints(run) :
	return list(zip(run[KEYS_POS.lat], run[KEYS_POS.lon]))

def is_in_map(lat, lon) :
	(left, bottom, right, top) = PARAM['map_bbox']
	return ((bottom < lat < top) and (left < lon < right))

## ===================== WORK :

def mapmatch_runs(runs) :

	# Road network (main graph component) with nearest-neighbor tree for the nodes
	g: nx.DiGraph
	(g, knn) = commons.inspect(('g', 'knn'))(
		pickle.load(open(PARAM['OSM_graph_file'], 'rb'))['main_component_with_knn']
	)

	# Nearest edges
	kne = (lambda q: graph.estimate_kne(g, knn, q, ke=20))

	mpl.use('Agg')
	import matplotlib.pyplot as plt

	def mm_callback(result) :
		fig: plt.Figure
		ax: plt.Axes

		if not ('plt' in result) :
			(fig, ax) = plt.subplots()
			result['plt'] = { 'fig' : fig, 'ax' : ax }

		if (result['status'] == "opti") :
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

		# Display/save figure here
		commons.makedirs(OFILE['progress_img'])
		fig.savefig(OFILE['progress_img'], bbox_inches='tight', pad_inches=0)

		# Next figure update
		result['nfu'] = dt.datetime.now() + dt.timedelta(seconds=2)

		# Note: need to close figure


	# Keep a certain distance between waypoints
	def sparsify(wps, dist=60) :
		a = next(iter(wps))
		yield a
		for b in wps :
			if (graph.geodist(a, b) >= dist) :
				a = b
				yield a

	for run in runs :

		waypoints = list(sparsify(run_waypoints(run)))
		if (len(waypoints) < 5) : continue

		print("waypoints:", waypoints)

		mapmatch_attempt = {
			k: run[k] for k in [KEYS.routeid, KEYS.dir, KEYS.runid, KEYS.busid]
		}

		mapmatch_attempt['EstPathUUID'] = uuid.uuid4().hex

		filename = OFILE['mapmatched'].format(routeid=mapmatch_attempt[KEYS.routeid], direction=mapmatch_attempt[KEYS.dir], estpathid=mapmatch_attempt['EstPathUUID'])

		#print("waypoints ({}): {}". format(len(waypoints), waypoints))

		try :
			commons.seed()
			result = graph.mapmatch(waypoints, g, kne, mm_callback, stubborn=0.2)
		except Exception as e :
			print("Mapmatch failed on run {} ({})".format(run[KEYS.runid], e))
			time.sleep(2)
			continue

		# Note: because figure is not jsonable cannot do
		# mapmatch_attempt['mapmatch_result'] = result

		for k in ['geo_path', 'path'] : mapmatch_attempt[k] = result[k]

		try :
			(fig, ax) = commons.inspect({'plt': ('fig', 'ax')})(result)
			commons.makedirs(filename)
			fig.savefig(filename + ".png", bbox_inches='tight', pad_inches=0)
			plt.close(fig)
		except Exception as e :
			print("Could not save figure {} ({})".format(filename + ".png", e))

		try :
			commons.makedirs(filename)
			with commons.logged_open(filename + ".json", 'w') as fd :
				json.dump(mapmatch_attempt, fd)
		except Exception as e :
			print("Failed to write mapmatch file {} ({})".format(filename + ".json", e))

		time.sleep(1)


def distill_shape(routeid, direction) :
	mpl.use('TkAgg')
	import matplotlib.pyplot as plt

	def remove_repeats(xx) :
		xx = list(xx)
		return [x for (x, y) in zip(xx, xx[1:]) if (x != y)] + xx[-1:]

	def commonest(xx) :
		xx = list(xx)
		return max(xx, key=xx.count)

	matchedruns_files = sorted(glob.glob(OFILE['mapmatched'].format(routeid=routeid, direction=direction, estpathid="*")))

	mms = [
		commons.zipjson_load(fn)
		for fn in matchedruns_files
	]

	from difflib import SequenceMatcher
	def pathsim(a, b) : return SequenceMatcher(None, a, b).ratio()

	# Number of map matched suggestions
	nmm = len(mms)
	nclusters = 3

	# Affinity matrix
	M = np.zeros((nmm, nmm))
	for ((i, mm1), (j, mm2)) in product(enumerate(mms), repeat=2) :
		M[i, j] = 1 - pathsim(remove_repeats(mm1['path']), remove_repeats(mm2['path']))

	from sklearn.cluster import AgglomerativeClustering
	labels = list(AgglomerativeClustering(linkage='complete', affinity='precomputed', n_clusters=nclusters).fit_predict(M))

	# Most common label -- largest cluster
	label1 = commonest(labels)

	print("Largest cluster size:", labels.count(label1))

	# Get the mapmatched paths corresponding to the largest cluster
	mms = [mm for (mm, label) in zip(mms, labels) if (label == label1)]

	geopaths = [[tuple(p) for p in remove_repeats(mm['geo_path'])] for mm in mms]
	#geo_path = [tuple(p) for p in ]

	(fig, ax) = plt.subplots()

	# Define start and end point for the route
	(a, b) = (commonest([gp[0] for gp in geopaths]), commonest([gp[-1] for gp in geopaths]))

	(y, x) = a
	ax.plot(x, y, 'o', c='g')

	(y, x) = b
	ax.plot(x, y, 'o', c='b')

	# pp = list(chain.from_iterable(geopaths))
	# pp = set([p for p in pp if (pp.count(p) > len(geopaths) / 2)])
	# geopaths = [
	# 	[p for p in gp if (p in pp)]
	# 	for gp in geopaths
	# ]

	plt.ion()
	plt.show()

	def get_route(geopaths) :
		geopaths = deepcopy(geopaths)

		def index_try(xx, x) :
			try :
				return xx.index(x)
			except ValueError :
				return None

		# Define start and end point for the route
		(p0, p1) = (commonest([gp[0] for gp in geopaths]), commonest([gp[-1] for gp in geopaths]))

		p = None
		while (len(geopaths) > 1) :
			p_old = p

			# The most frequent candidate-location
			p = commonest(gp[0] for gp in geopaths)

			# Where does it appear in other paths?
			for (i, m) in enumerate(index_try(gp, p) for gp in geopaths) :
				if (m is not None) :
					geopaths[i] = geopaths[i][(m + 1):]

			# Remove emptied geopaths
			geopaths = [gp for gp in geopaths if gp]

			if (p == p0) : p0 = None # The start of the route

			# Have met the start, but not past the end of route
			# Do not trust a weak suggestion
			if (not p0) :

				yield p

				# if (p_old):
				# 	(y, x) = zip(p, p_old)
				# 	ax.plot(x, y, '.-')
				# 	plt.pause(0.1)

			if (p == p1) : break

	route = list(get_route(geopaths))

	(y, x) = zip(*route)
	ax.plot(x, y, 'r-')

	plt.ioff()
	plt.show()

	return route

def map_routes() :
	routeid_of = (lambda r: r[KEYS.routeid])
	direction_of = (lambda r: r[KEYS.dir])
	run_id = (lambda r : (routeid_of(r), direction_of(r)))

	# List of filenames, one file per physical bus, identified by plate number
	bus_files = sorted(glob.glob(IFILE['busses'].format(busid="*")))

	runs = list(
		run
		for fn in bus_files
		for run in commons.zipjson_load(fn)
	)

	print("Found {} runs".format(len(runs)))

	runs = [run for run in runs if all(is_in_map(*p) for p in run_waypoints(run))]

	print("Found {} runs inside the map".format(len(runs)))

	runs = { k : list(g) for (k, g) in groupby(sorted(runs, key=run_id), run_id) }
	#runs = commons.index_dicts_by_key(runs, run_id))

	# Sort the route-directions by decreasing number of measurements
	for (case, runs) in sorted(runs.items(), key=(lambda cr : -len(cr[1]))) :
		print("Route {}, direction {} ({} runs)".format(*case, len(runs)))

		if (len(runs) <= 4) :
			print("Aborting: too few runs.")
			break

		mapmatch_runs(runs)
		route = distill_shape(*case)

		print(route)


## ===================== PLAY :

pass


## ================== OPTIONS :

OPTIONS = {
	'MAP_ROUTES' : map_routes,
}

## ==================== ENTRY :

if (__name__ == "__main__") :
	commons.parse_options(OPTIONS)
