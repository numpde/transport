#!/usr/bin/python3

# RA, 2018-10-31

## ================== IMPORTS :

import commons

import time
import json
import glob
import inspect
from itertools import chain
import matplotlib.pyplot as plt


## ==================== NOTES :

pass


## ==================== INPUT :

IFILE = {
	'busses' : "OUTPUT/13/Kaohsiung/UV/{busid}.json",

	'route-stops' : "ORIGINALS/MOTC/Kaohsiung/CityBusApi_StopOfRoute/data.json",
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

	route_stops = commons.index_dicts_by_key(commons.zipjson_load(IFILE['route-stops']), (lambda r: r['SubRouteUID']))

	bus_files = sorted(glob.glob(IFILE['busses'].format(busid="*")))
	fn = bus_files[1]

	runs = commons.zipjson_load(fn)

	for run in runs :

		print(run)

		route_uid = run['SubRouteUID']

		route = route_stops[route_uid]
		stops = dict(zip(route['Direction'], route['Stops']))[run['Direction']]

		# Clear figure
		plt.clf()

		# Draw stops
		for stop in stops :
			p = stop['StopPosition']
			(y, x) = (p['PositionLat'], p['PositionLon'])
			plt.scatter(x, y, c='b', marker='o' )

		# Trace bus
		(y, x) = (run['PositionLat'], run['PositionLon'])
		plt.plot(x, y, '-', c='r', linewidth=1)

		plt.draw(); plt.pause(0.1)

		plt.ioff()
		plt.show()

	return

## ================== OPTIONS :

OPTIONS = {
	'VIS1' : vis1,
}

## ==================== ENTRY :

if (__name__ == "__main__") :

	assert(commons.parse_options(OPTIONS))
