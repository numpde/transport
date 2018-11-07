#!/usr/bin/python3

# RA, 2018-10-31

## ================== IMPORTS :

from helpers import commons

import glob
import inspect
from collections import defaultdict
import matplotlib.pyplot as plt


## ==================== NOTES :

pass


## ==================== INPUT :

IFILE = {
	'busses' : "OUTPUT/13/Kaohsiung/UV/{busid}.json",

	'route-stops' : "OUTPUT/00/ORIGINAL_MOTC/Kaohsiung/CityBusApi_StopOfRoute/data.json",

	'OSM': "OUTPUT/02/UV/kaohsiung.pkl",
}


## =================== OUTPUT :

OFILE = {
}


## ==================== PARAM :

PARAM = {
}

## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))


## ===================== WORK :

pass

## ===================== PLAY :

# Small visualization of the bus record
# Follow one bus
def vis1() :

	# OSM = pickle.load(open(IFILE['OSM'], 'rb'))
	# for (route_id, route) in OSM['rels']['route'].items():
	# 	# Skip non-bus routes
	# 	if not (route['t'].get('route') == 'bus'): continue
	#
	# 	route_name = route['t'].get('name')
	#
	# 	route_ref = route['t']['ref']
	# 	#if (route_ref == '88') :
	# 	print(route_name, route_id, route['t'])
	# exit(39)

	route_stops = commons.index_dicts_by_key(commons.zipjson_load(IFILE['route-stops']), (lambda r: r['SubRouteUID']))

	runs_by_route = defaultdict(list)

	bus_files = sorted(glob.glob(IFILE['busses'].format(busid="*")))

	for fn in bus_files :
		runs = commons.zipjson_load(fn)
		for run in runs :
			runs_by_route[run['SubRouteUID']].append(run)

	route_uid = 'KHH122'

	runs = runs_by_route[route_uid]
	route = route_stops[route_uid]

	for run in runs :

		run_dir = run['Direction']
		stops = dict(zip(route['Direction'], route['Stops']))[run_dir]

		# Clear figure
		plt.ion()

		# Draw stops
		for stop in stops :
			p = stop['StopPosition']
			(y, x) = (p['PositionLat'], p['PositionLon'])
			plt.scatter(x, y, c=('b' if (run_dir == 0) else 'g'), marker='o')

		# Trace bus
		(y, x) = (run['PositionLat'], run['PositionLon'])
		h = plt.plot(x, y, '--+', c='r', linewidth=1)

		plt.draw(); plt.pause(1)

		h[0].remove()

		#plt.ioff()
		plt.show()

	return

## ================== OPTIONS :

OPTIONS = {
	'VIS1' : vis1,
}

## ==================== ENTRY :

if (__name__ == "__main__") :

	assert(commons.parse_options(OPTIONS))
