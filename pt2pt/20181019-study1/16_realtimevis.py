#!/usr/bin/python3

# RA, 2018-10-31

## ================== IMPORTS :

from helpers import commons
from helpers import maps

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

	'mapbox_api_token' : open(".credentials/UV/mapbox-token.txt", 'r').read(),
}

## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))


## ===================== WORK :

pass

## ===================== PLAY :

# At what time does a given bus visit the bus stops?
def bus_at_stops(run, stops) :
	# These are sparse samples of a bus trajectory
	candidate_gps = list(zip(run['PositionLat'], run['PositionLon']))

	# These are fixed
	reference_gps = list(commons.inspect({'StopPosition': ('PositionLat', 'PositionLon')})(stop) for stop in stops)

	print(candidate_gps)
	print(reference_gps)

	exit(39)
	pass

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

	routeid_of = (lambda r: r['SubRouteUID'])

	# List of filenames, one file per physical bus, identified by plate number
	bus_files = sorted(glob.glob(IFILE['busses'].format(busid="*")))

	# Refile bus runs by their route ID
	runs_by_route = defaultdict(list)
	for fn in bus_files :
		runs = commons.zipjson_load(fn)
		for run in runs :
			runs_by_route[routeid_of(run)].append(run)

	#
	route_stops = commons.index_dicts_by_key(commons.zipjson_load(IFILE['route-stops']), routeid_of)

	# Are those valid route ID that can be found among the routes?
	unknown_route_ids = sorted(set(runs_by_route.keys()) - set(route_stops.keys()))

	if unknown_route_ids :
		print("The following route IDs from bus records are unknown:")
		print(", ".join(unknown_route_ids))
		raise KeyError("Unkown route IDs in bus records")

	#

	route_uid = 'KHH122'

	runs = runs_by_route[route_uid]
	route = route_stops[route_uid]

	# Kaohsiung (left, bottom, right, top)
	bbox = (120.2593, 22.5828, 120.3935, 22.6886)
	(left, bottom, right, top) = bbox

	# Download the background map
	i = maps.get_map_by_bbox(bbox, token=PARAM['mapbox_api_token'])

	# Show the background map
	plt.ion()
	plt.gca().axis([left, right, bottom, top])
	plt.gca().imshow(i, extent=(left, right, bottom, top), interpolation='quadric')
	plt.show()


	stops_by_direction = dict(zip(route['Direction'], route['Stops']))

	# Draw stops for both route directions
	for (dir, stops) in stops_by_direction.items() :

		# Stop locations
		(y, x) = zip(*[
			commons.inspect({'StopPosition': ('PositionLat', 'PositionLon')})(stop)
			for stop in stops
		])

		# Plot as dots
		plt.scatter(x, y, c=('b' if dir else 'g'), marker='o', s=4)


	# Show bus location

	for run in runs :

		# Trace bus
		(y, x) = (run['PositionLat'], run['PositionLon'])
		h = plt.plot(x, y, '--+', c='r', linewidth=1)

		plt.draw()
		plt.savefig("{}.png".format(route_uid), dpi=180)
		plt.pause(1)

		h[0].remove()

		bus_at_stops(run, stops_by_direction[run['Direction']])

	return

## ================== OPTIONS :

OPTIONS = {
	'VIS1' : vis1,
}

## ==================== ENTRY :

if (__name__ == "__main__") :
	commons.parse_options(OPTIONS)
