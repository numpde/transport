#!/usr/bin/python3

# RA, 2018-11-15

## ================== IMPORTS :

from helpers import commons, maps, graph

import re
import uuid
import time
import json
import networkx as nx
import math
import random
import pickle
import inspect
import traceback
import datetime as dt
import sklearn.neighbors

import matplotlib as mpl
# Note: do not import pyplot here -- may need to select renderer


## ================= FILE I/O :

open = commons.logged_open


## ==================== INPUT :

IFILE = {
	'OSM_graph_file' : "OUTPUT/02/UV/kaohsiung.pkl",

	'segment_by_route' : "OUTPUT/13/{scenario}/byroute/UV/{routeid}-{dir}.json",
}


## =================== OUTPUT :

OFILE = {
	'progress_img': "OUTPUT/14/progress/UV/current_route.png",
	'progress_txt': "OUTPUT/14/progress/UV/current_route.txt",
	'progress_gpx': "OUTPUT/14/progress/UV/current_route.gpx",

	'mapmatched': "OUTPUT/14/mapmatched/{scenario}/{routeid}-{direction}/UV/{mapmatch_uuid}.{ext}",
}

commons.makedirs(OFILE)


## ==================== PARAM :

PARAM = {
	'mapbox_api_token' : open(".credentials/UV/mapbox-token.txt", 'r').read(),

	# Only retain routes contained in this area (left, bottom, right, top)
	# Will be filled based on the graph if 'None'
	'graph_bbox' : None,
	# Example:
	#'graph_bbox' : (120.2593, 22.5828, 120.3935, 22.6886),

	'min_run_waypoints' : 4,

	'min_runs_to_mapmatch' : 24,
	'max_runs_to_mapmatch' : 24,

	'waypoints_min_distance' : 60, # (meters)
}


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

	busstatus = 'BusStatus'
	dutystatus = 'DutyStatus'


# Helpers
BUSID_OF = (lambda b: b[KEYS.busid])
ROUTEID_OF = (lambda r: r[KEYS.routeid])
DIRECTION_OF = (lambda r: r[KEYS.dir])

# What finally identifies a one-way route
RUN_KEY = (lambda r : (ROUTEID_OF(r), DIRECTION_OF(r)))


## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))

def run_waypoints(run) :
	return list(map(tuple, run[KEYS.pos]))

# Keep a certain distance between waypoints (in meters)
def sparsify(wps, dist=PARAM['waypoints_min_distance']) :
	a = next(iter(wps))
	yield a
	for b in wps :
		if (graph.geodist(a, b) >= dist) :
			a = b
			yield a

def compute_graph_bbox() :
	g: nx.DiGraph = pickle.load(open(IFILE['OSM_graph_file'], 'rb'))['main_component_with_knn']['g']
	(left, bottom, right, top) = [
		min(lon for (n, (lat, lon)) in g.nodes.data('pos')),
		min(lat for (n, (lat, lon)) in g.nodes.data('pos')),
		max(lon for (n, (lat, lon)) in g.nodes.data('pos')),
		max(lat for (n, (lat, lon)) in g.nodes.data('pos'))
	]
	return (left, bottom, right, top)

def is_in_map(lat, lon) :
	(left, bottom, right, top) = PARAM['graph_bbox']
	return ((bottom < lat < top) and (left < lon < right))



## ==================== SLAVE :

def mapmatch_runs(scenario, runs) :

	# Road network (main graph component) with nearest-neighbor tree for the nodes
	g: nx.DiGraph
	knn : sklearn.neighbors.NearestNeighbors
	(g, knn) = commons.inspect({'main_component_with_knn': ('g', 'knn')})(
		pickle.load(open(IFILE['OSM_graph_file'], 'rb'))
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

		ax.set_title("{} ({}%)".format(result['status'], math.floor(100 * result.get('progress', 0))))

		if ('waypoints' in result) :
			(y, x) = zip(*result['waypoints'])
			ax.plot(x, y, 'o', c='m', markersize=4)

		if ('geo_path' in result) :
			(y, x) = zip(*result['geo_path'])
			ax.plot(x, y, 'b--', linewidth=2, zorder=-50)

		# Display/save figure here
		commons.makedirs(OFILE['progress_img'])
		with open(OFILE['progress_img'], 'wb') as fd :
			fig.savefig(fd, bbox_inches='tight', pad_inches=0)

		# Next figure update
		result['nfu'] = dt.datetime.now() + dt.timedelta(seconds=2)

		# Log into a GPX file
		with open(OFILE['progress_gpx'], 'w') as fd :
			fd.write(graph.simple_gpx(result['waypoints'], [result.get('geo_path', [])]).to_xml())

		# Note: need to close figure


	for run in runs :

		waypoints = list(sparsify(run_waypoints(run)))

		if (len(waypoints) < PARAM['min_run_waypoints']) :
			print("Warning: too few waypoints -- skipping.")
			continue

		print("Waypoints:", waypoints)

		mapmatch_attempt = {
			k: run[k] for k in [KEYS.routeid, KEYS.dir, KEYS.runid, KEYS.busid]
		}

		mapmatch_attempt['MapMatchUUID'] = uuid.uuid4().hex

		fn = OFILE['mapmatched'].format(scenario=scenario, routeid=mapmatch_attempt[KEYS.routeid], direction=mapmatch_attempt[KEYS.dir], mapmatch_uuid=mapmatch_attempt['MapMatchUUID'], ext="{ext}")

		#print("waypoints ({}): {}". format(len(waypoints), waypoints))

		try :
			commons.seed()
			result = graph.mapmatch(waypoints, g, kne, mm_callback, stubborn=0.2)
		except Exception as e :
			print("Warning: Mapmatch failed on run {} ({})".format(run[KEYS.runid], e))
			print(traceback.format_exc())
			time.sleep(2)
			continue

		# Copy relevant fields from the mapmatcher result
		for k in ['waypoints', 'path', 'geo_path', 'mapmatcher_version'] :
			mapmatch_attempt[k] = result[k]

		# Save the result in different formats, in this directory
		commons.makedirs(fn.format(ext='~~~'))

		#  o) Image

		try :
			fig = result['plt']['fig']
			with open(fn.format(ext="png"), 'wb') as fd :
				fig.savefig(fd, bbox_inches='tight', pad_inches=0)
			plt.close(fig)
		except Exception as e :
			print("Warning: Could not save figure {} ({})".format(fn.format(ext="png"), e))

		#  o) JSON

		try :
			with open(fn.format(ext="json"), 'w') as fd :
				json.dump(mapmatch_attempt, fd)
		except Exception as e :
			print("Warning: Failed to write mapmatch file {} ({})".format(fn.format(ext="json"), e))

		#  o) GPX

		try :
			with open(fn.format(ext="gpx"), 'w') as fd :
				fd.write(graph.simple_gpx(mapmatch_attempt['waypoints'], [mapmatch_attempt['geo_path']]).to_xml())
		except Exception as e :
			print("Warning: Failed to write GPX file {} ({})".format(fn.format(ext="gpx"), e))

		time.sleep(1)


## =================== MASTER :

def mapmatch_all() :

	commons.seed()

	PARAM['graph_bbox'] = compute_graph_bbox()

	route_files = commons.ls(IFILE['segment_by_route'].format(scenario="**", routeid="*", dir="*"))
	print("Found {} route files.".format(len(route_files)))

	for route_file in route_files :
		print("===")
		print("Analyzing route file {}.".format(route_file))

		(scenario, routeid, dir) = re.fullmatch(IFILE['segment_by_route'].format(scenario="(.*)", routeid="(.*)", dir="(.*)"), route_file).groups()
		dir = int(dir)
		print("Route: {}, direction: {} (from scenario: {})".format(routeid, dir, scenario))

		# Load all bus run segments for this case
		runs = commons.zipjson_load(route_file)
		print("Number of runs: {} ({})".format(len(runs), "total"))

		# Check that the file indeed contains only one type of route
		assert({(routeid, dir)} == set(RUN_KEY(r) for r in runs))

		# Keep only runs within the map
		runs = [run for run in runs if all(is_in_map(*p) for p in run[KEYS.pos])]
		print("Number of runs: {} ({})".format(len(runs), "within the map bbox"))

		# Q: clustering here?

		if not runs :
			print("File does not contain usable runs.")
			continue

		# Existing mapmatched runs for this route
		def get_mapmatched_files() :
			return commons.ls(OFILE['mapmatched'].format(scenario=scenario, routeid=routeid, direction=dir, mapmatch_uuid="*", ext="json"))

		try :

			if get_mapmatched_files() :
				print("Skipping mapmatch: mapmatched files found.")
				continue

			if (len(runs) > PARAM['max_runs_to_mapmatch']) :
				print("Out of {} available runs, will mapmatch only random {}.".format(len(runs), PARAM['max_runs_to_mapmatch']))
				runs = commons.random_subset(runs, k=PARAM['max_runs_to_mapmatch'])

			if (len(runs) < PARAM['min_runs_to_mapmatch']) :
				print("Skipping mapmatch: too few runs.")
				continue

			mapmatch_runs(scenario, runs)

		except Exception as e:
			print("Mapmatch failed ({}).".format(e))
			print(traceback.format_exc())


## ==================== ENTRY :

if (__name__ == "__main__") :
	mapmatch_all()
