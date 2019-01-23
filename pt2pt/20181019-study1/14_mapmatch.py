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
import pickle
import inspect
import traceback
import datetime as dt
import sklearn.neighbors

import matplotlib as mpl
# Note: do not import pyplot here -- may need to select renderer

# from typing import Tuple

## ================= FILE I/O :

open = commons.logged_open


## ==================== INPUT :

IFILE = {
	'OSM_graph_file' : "OUTPUT/02/UV/kaohsiung.pkl",

	'segment_by_route' : [
		# "ORIGINALS/13/{scenario}/byroute/{routeid}-{direction}.json",
		"OUTPUT/13/{scenario}/byroute/UV/{routeid}-{direction}.json",
	]
}


## =================== OUTPUT :

OFILE = {
	'progress': "OUTPUT/14/progress/UV/current_route.{ext}",

	'mapmatched': "OUTPUT/14/mapmatched/{scenario}/{routeid}-{direction}/UV/{mapmatch_uuid}.{ext}",
}

commons.makedirs(OFILE)


## ==================== PARAM :

PARAM = {
	'mapbox_api_token' : commons.token_for('mapbox'),

	# Only retain routes contained in this area (left, bottom, right, top)
	# Will be filled based on the graph if 'None'
	'graph_bbox' : None,
	# Example:
	#'graph_bbox' : (120.2593, 22.5828, 120.3935, 22.6886),

	'min_runs_to_mapmatch' : 6,
	'max_runs_to_mapmatch' : 128,
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

def is_in_map(lat, lon) :
	(left, bottom, right, top) = PARAM['graph_bbox']
	return ((bottom < lat < top) and (left < lon < right))

def trim_graph_to_busable(g: nx.DiGraph) :
	commons.logger.warning("trim_graph_to_busable not implemented")
	return g

## ==================== SLAVE :

def mapmatch_runs(scenario, runs) :

	# Road network (main graph component) with nearest-neighbor tree for the nodes
	g: nx.DiGraph

	(g, knn) = commons.inspect(['g', 'knn'])(
		pickle.load(open(IFILE['OSM_graph_file'], 'rb'))['main_component_with_knn']
	)

	g = trim_graph_to_busable(g)

	# Nearest edges computer
	kne = (lambda q: graph.estimate_kne(g, knn, q, ke=20))

	mpl.use('Agg')
	mpl.rcParams['figure.max_open_warning'] = 100

	import matplotlib.pyplot as plt

	#
	def make_figure(result) -> dict :
		ax: plt.Axes

		if ('plt' in result) :
			(fig, ax) = commons.inspect({'plt' : ('fig', 'ax')})(result)
		else :
			(fig, ax) = plt.subplots()
			result['plt'] = { 'fig' : fig, 'ax' : ax }
			result['auto_close_fig'] = commons.UponDel(lambda : plt.close(fig))

		ax.cla()

		ax.set_title("{} ({}%)".format(result['status'], math.floor(100 * result.get('progress', 0))))

		if ('waypoints_all' in result) :
			(y, x) = zip(*result['waypoints_all'])
			ax.plot(x, y, 'o', c='m', markersize=2)

		if ('geo_path' in result) :
			(y, x) = zip(*result['geo_path'])
			ax.plot(x, y, 'b--', linewidth=2, zorder=100)

		return result['plt']

	#
	def mm_callback(result) -> None:

		if (result['status'] == "opti") :
			if (dt.datetime.now() < result.get('nfu', dt.datetime.min)) :
				return

		# Log into a GPX file
		if ('waypoints_all' in result) :
			with open(OFILE['progress'].format(ext="gpx"), 'w') as fd :
				fd.write(graph.simple_gpx(result['waypoints_all'], [result.get('geo_path', [])]).to_xml())

		# Save figure
		make_figure(result)
		with open(OFILE['progress'].format(ext="png"), 'wb') as fd :
			fig.savefig(fd, bbox_inches='tight', pad_inches=0)

		# Next figure update
		result['nfu'] = dt.datetime.now() + dt.timedelta(seconds=5)


	# Collect all bus runs
	runs_by_runid = {
		run[KEYS.runid] : run
		for run in runs
	}

	# Collect all waypoints
	waypoints_by_runid = {
		runid : list(map(tuple, run[KEYS.pos]))
		for (runid, run) in runs_by_runid.items()
	}

	commons.logger.debug(json.dumps(runs, indent=2))

	commons.logger.info("Running mapmatch on {} runs".format(len(runs)))

	# MAPMATCH RUNS
	results = graph.mapmatch(waypoints_by_runid, g, kne, knn=knn, callback=None, stubborn=0.2, many_partial=True)

	for result in results :

		commons.logger.info("Got mapmatch with waypoints {}".format(result['waypoints_used']))

		# The run on which mapmatch operated
		run = runs_by_runid[result['waypoint_setid']]

		# Collect initial info about the mapmatch attempt
		mapmatch_attempt = {
			k: run[k] for k in [KEYS.routeid, KEYS.dir, KEYS.runid, KEYS.busid]
		}

		# Attach a unique identifier for this mapmatch
		mapmatch_attempt['MapMatchUUID'] = uuid.uuid4().hex

		# Filename without the extension
		fn = OFILE['mapmatched'].format(scenario=scenario, routeid=mapmatch_attempt[KEYS.routeid], direction=mapmatch_attempt[KEYS.dir], mapmatch_uuid=mapmatch_attempt['MapMatchUUID'], ext="{ext}")

		# Copy relevant fields from the mapmatcher result
		for k in ['waypoints_used', 'path', 'geo_path', 'mapmatcher_version'] :
			mapmatch_attempt[k] = result[k]

		# Save the result in different formats, in this directory
		commons.makedirs(fn.format(ext='~~~'))

		#  o) Image

		try :
			# Make and save the figure
			fig = make_figure(result)['fig']
			with open(fn.format(ext="png"), 'wb') as fd :
				fig.savefig(fd, bbox_inches='tight', pad_inches=0)
		except Exception as e :
			commons.logger.warning("Could not save figure {} ({})".format(fn.format(ext="png"), e))

		#  o) JSON

		try :
			with open(fn.format(ext="json"), 'w') as fd :
				json.dump(mapmatch_attempt, fd)
		except Exception as e :
			commons.logger.warning("Failed to write mapmatch file {} ({})".format(fn.format(ext="json"), e))

		#  o) GPX

		try :
			with open(fn.format(ext="gpx"), 'w') as fd :
				fd.write(graph.simple_gpx(mapmatch_attempt['waypoints_used'], [mapmatch_attempt['geo_path']]).to_xml())
		except Exception as e :
			commons.logger.warning("Failed to write GPX file {} ({})".format(fn.format(ext="gpx"), e))

		time.sleep(1)


## =================== MASTER :

def mapmatch_all() :

	commons.seed()

	PARAM['graph_bbox'] = maps.bbox_for_points(
		nx.get_node_attributes(
			trim_graph_to_busable(pickle.load(open(IFILE['OSM_graph_file'], 'rb'))['main_component_with_knn']['g']),
			'pos'
		).values()
	)

	for route_file_template in IFILE['segment_by_route']:

		route_files = commons.ls(route_file_template.format(scenario="**", routeid="*", direction="*"))

		commons.logger.info("Route file template: {}".format(route_file_template))
		commons.logger.info("Found {} route files".format(len(route_files)))

		for route_file in route_files :
			# time.sleep(2)

			commons.logger.info("===")
			commons.logger.info("Analyzing route file {}.".format(route_file))

			case = commons.unformat(route_file_template, route_file)
			commons.logger.info("Route: {routeid}, direction: {direction} (from scenario: {scenario})".format(**case))

			# # DEBUG
			# if not ("KHH239-0" == "{routeid}-{direction}".format(**case)) :
			# 	continue

			# Load all bus run segments for this case
			runs = commons.zipjson_load(route_file)
			commons.logger.info("Number of runs: {} ({})".format(len(runs), "total"))

			# Check that the file indeed contains only one type of route
			assert({(case['routeid'], int(case['direction']))} == set(RUN_KEY(r) for r in runs))

			# Remove runs that have a negative quality flag
			runs = [run for run in runs if not (run.get('quality') == "-")]
			commons.logger.info("Number of runs: {} ({})".format(len(runs), "not marked as bad quality"))

			# Keep only runs within the map
			runs = [run for run in runs if all(is_in_map(*p) for p in run[KEYS.pos])]
			commons.logger.info("Number of runs: {} ({})".format(len(runs), "within the map bbox"))

			if (len(runs) > PARAM['max_runs_to_mapmatch']) :
				commons.logger.info("Out of {} available runs, will mapmatch only random {}".format(len(runs), PARAM['max_runs_to_mapmatch']))
				runs = commons.random_subset(runs, k=PARAM['max_runs_to_mapmatch'])

			if (len(runs) < PARAM['min_runs_to_mapmatch']) :
				commons.logger.warning("Skipping mapmatch: too few runs.")
				continue

			# Q: clustering here?

			# Existing mapmatched runs for this route
			existing = commons.ls(OFILE['mapmatched'].format(**case, mapmatch_uuid="*", ext="json"))

			if existing :
				commons.logger.warning("Skipping mapmatch: {} mapmatched files found".format(len(existing)))
				continue

			try :

				mapmatch_runs(case['scenario'], runs)

			except Exception as e :

				commons.logger.error("Mapmatch failed ({}) \n{}".format(e, traceback.format_exc()))
				commons.logger.warning("Mapmatch incomplete on route {routeid}-{direction} from scenario '{scenario}'".format(**case))
				time.sleep(5)


## ==================== ENTRY :

if (__name__ == "__main__") :
	mapmatch_all()
