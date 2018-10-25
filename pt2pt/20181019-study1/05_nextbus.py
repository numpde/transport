#!/usr/bin/python3

# RA, 2018-10-25

## ================== IMPORTS :

import os
import json
import inspect
import glob


## ==================== NOTES :

pass


## ==================== INPUT :

IFILE = {
	'route-json' : "OUTPUT/04/kaohsiung_bus_routes/route_{route_id}.json",
}


## =================== OUTPUT :

OFILE = {
	'stops-json' : "OUTPUT/05/kaohsiung_bus_stops.json",
}

# Create output directories
for f in OFILE.values() : os.makedirs(os.path.dirname(f), exist_ok=True)


## ==================== PARAM :

PARAM = {
	'' : 0,
}

## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))

# Log which files are opened
def logged_open(filename, mode='r', *argv, **kwargs) :
	print("({}):\t{}".format(mode, filename))
	return open(filename, mode, *argv, **kwargs)


## ===================== WORK :

def process() :

	stops = dict()

	for filename in glob.glob(IFILE['route-json'].format(route_id="*")) :

		J = json.load(logged_open(filename, 'r'))

		for stop in J['zh'] :
			sid = stop['SID']
			if not (sid in stops) :
				stops[sid] = {
					'lat' : stop['Latitude'],
					'lon' : stop['Longitude'],
					'routes' : set()
				}
			stops[sid]['routes'].add(stop['RouteId'])

		for lang in J.keys() :
			for stop in J[lang] :
				stops[stop['SID']][lang] = stop['NameZh']

	for sid in stops.keys() :
		stops[sid]['routes'] = sorted(stops[sid]['routes'])

	json.dump(stops, logged_open(OFILE['stops-json'], 'w'))

## ==================== ENTRY :

if (__name__ == "__main__") :
	process()

