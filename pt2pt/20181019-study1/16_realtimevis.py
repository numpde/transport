#!/usr/bin/python3

# RA, 2018-10-31

## ================== IMPORTS :

from helpers import commons
from helpers import maps
from helpers.graph import dist_to_segment

import numpy as np
import dateutil
import datetime as dt
import inspect
import geopy.distance
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

	# TODO: some candidate_tdt have large time-gaps

	# These are sparse samples of a bus trajectory
	candidate_gps = list(zip(run['PositionLat'], run['PositionLon']))
	# Timestamps of GPS records as datetime objects
	candidate_tdt = list(map(dateutil.parser.parse, run['GPSTime']))

	# These are fixed platform locations
	reference_gps = list(commons.inspect({'StopPosition': ('PositionLat', 'PositionLon')})(stop) for stop in stops)

	# Goal: obtain an estimate ref_guess_tdt of when the bus is nearest to the platforms

	print(candidate_gps)
	print(run['GPSTime'])
	print(candidate_tdt)
	print(reference_gps)

	segments = list(zip(candidate_gps[:-1], candidate_gps[1:]))
	segm_tdt = list(zip(candidate_tdt[:-1], candidate_tdt[1:]))

	M = np.vstack([dist_to_segment(r, s)[0] for s in segments] for r in reference_gps)

	if not M.size : return

	# M contains distances in meters
	# We expect the bus to travel about 17km/h on average, say 5m/s
	# So, a penalty of p = 0.1m/s should not make much of a difference,
	# unless the bus is idling

	# Penalty rate
	p = 0.1 # m/s
	# Penalty matrix
	P = np.vstack(len(reference_gps) * [np.cumsum([p * (t1 - t0).seconds for (t0, t1) in segm_tdt])])

	# Add idling penalty
	M += P

	match = commons.align(M)
	print(match)

	segments = [segments[j] for j in match]
	segm_tdt = [segm_tdt[j] for j in match]
	seg_dist = [dist_to_segment(r, s) for (r, s) in zip(reference_gps, segments)]

	ref_guess_tdt = [
		t0 + q * (t1 - t0)
		for ((d, q), (t0, t1)) in zip(seg_dist, segm_tdt)
	]

	for t in ref_guess_tdt : print(t)

	is_monotone = all((s <= t) for (s, t) in zip(ref_guess_tdt[:-1], ref_guess_tdt[1:]))
	print("Monotonicity:", is_monotone)

	(fig, ax) = plt.subplots()
	ax.imshow(M)

	while plt.fignum_exists(fig.number) :
		try :
			plt.pause(0.1)
		except :
			break

	plt.close(fig)

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
	bus_files = commons.ls(IFILE['busses'].format(busid="*"))

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

	route_uid = 'KHH24'

	runs = runs_by_route[route_uid]
	route = route_stops[route_uid]

	# Kaohsiung (left, bottom, right, top)
	bbox = (120.2593, 22.5828, 120.3935, 22.6886)
	(left, bottom, right, top) = bbox

	# Download the background map
	i = maps.get_map_by_bbox(bbox, token=PARAM['mapbox_api_token'])

	# Show the background map
	(fig, ax) = plt.subplots()
	plt.ion()
	ax.axis([left, right, bottom, top])
	ax.imshow(i, extent=(left, right, bottom, top), interpolation='quadric')

	#fig.canvas.draw_idle()

	plt.pause(0.1)


	stops_by_direction = dict(zip(route['Direction'], route['Stops']))

	# Draw stops for both route directions
	for (dir, stops) in stops_by_direction.items() :

		# Stop locations
		(y, x) = zip(*[
			commons.inspect({'StopPosition': ('PositionLat', 'PositionLon')})(stop)
			for stop in stops
		])

		# Plot as dots
		ax.scatter(x, y, c=('b' if dir else 'g'), marker='o', s=4)


	# Show bus location

	for run in runs :

		# Trace bus
		(y, x) = (run['PositionLat'], run['PositionLon'])
		h1 = ax.plot(x, y, '--+', c='r', linewidth=1)
		h2 = ax.plot(x[0], y[0], 'o', c='r')
		h3 = ax.plot(x[-1], y[-1], 's', c='r')

		plt.title(run['PlateNumb'])

		#plt.savefig("{}.png".format(route_uid), dpi=180)
		plt.pause(0.1)

		bus_at_stops(run, stops_by_direction[run['Direction']])

		plt.pause(0.1)
		[h[0].remove() for h in [h1, h2, h3]]

	return

## ================== OPTIONS :

OPTIONS = {
	'VIS1' : vis1,
}

## ==================== ENTRY :

if (__name__ == "__main__") :
	commons.parse_options(OPTIONS)
