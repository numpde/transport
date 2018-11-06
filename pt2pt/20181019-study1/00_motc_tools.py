#!/usr/bin/python3

# RA, 2018-11-06

## ================== IMPORTS :

import commons

import gpxpy, gpxpy.gpx
import re
import difflib
import pickle
import json
import glob
import time
import inspect
from itertools import chain
from collections import defaultdict

# import matplotlib.pyplot as plt


## ==================== NOTES :

pass

## ==================== INPUT :

IFILE = {
	'MOTC_routes': "ORIGINALS/MOTC/Kaohsiung/CityBusApi_StopOfRoute/data.json",
	'MOTC_shapes': "ORIGINALS/MOTC/Kaohsiung/CityBusApi_Shape/data.json",
	#'MOTC_stops': "ORIGINALS/MOTC/Kaohsiung/CityBusApi_Stop/data.json",
}

## =================== OUTPUT :

OFILE = {
	'GPX_shape' : "OUTPUT/00/GPX/Kaohsiung/UV/route_{route_id}_{dir}_shape.gpx",
	'GPX_stops' : "OUTPUT/00/GPX/Kaohsiung/UV/route_{route_id}_{dir}_stops.gpx",
}

commons.makedirs(OFILE)

## ==================== PARAM :

PARAM = {
}

## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))


# Print a JSON nicely
def pretty_print(J):
	print(json.dumps(J, indent=2, ensure_ascii=False))


## ===================== WORK :

def write_route_shape() :

	#motc_stops = commons.index_dicts_by_key(commons.zipjson_load(IFILE['MOTC_stops']), (lambda r: r['StopUID']))

	# Note: index by RouteUID (not SubRouteUID) because...
	motc_routes = commons.index_dicts_by_key(commons.zipjson_load(IFILE['MOTC_routes']), (lambda r: r['RouteUID']), preserve_singletons=['Direction', 'Stops'])
	# ...this file only provides a RouteUID for each record
	motc_shapes = commons.zipjson_load(IFILE['MOTC_shapes'])

	for shape in motc_shapes :

		(route_id, dir) = (shape['RouteUID'], shape['Direction'])

		route = motc_routes[route_id]

		# Parse LINESTRING
		(lon, lat) = zip(*re.findall(r'(?P<lon>[0-9.]+)[ ](?P<lat>[0-9.]+)', shape['Geometry']))
		(lon, lat) = (list(map(float, lon)), list(map(float, lat)))

		gpx = gpxpy.gpx.GPX()

		if dir in route['Direction'] :
			for stop in dict(zip(route['Direction'], route['Stops']))[dir] :
				(p, q) = (stop['StopPosition']['PositionLat'], stop['StopPosition']['PositionLon'])
				stop_name = "{}-#{}: {} / {}".format(dir, stop['StopSequence'], stop['StopName']['Zh_tw'], stop['StopName']['En'])
				stop_desc = "{} ({})".format(stop['StopUID'], stop['StationID'])
				gpx.waypoints.append(gpxpy.gpx.GPXWaypoint(latitude=p, longitude=q, name=stop_name, description=stop_desc))
		else :
			print("Route {}, direction {} not found in MOTC_routes".format(route_id, dir))

		# Create first track in our GPX
		gpx_track = gpxpy.gpx.GPXTrack()
		gpx.tracks.append(gpx_track)

		# Create first segment in our GPX track
		gpx_segment = gpxpy.gpx.GPXTrackSegment()
		gpx_track.segments.append(gpx_segment)

		# Create points
		for (p, q) in zip(lat, lon) :
			gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=p, longitude=q))

		fn = OFILE['GPX_shape'].format(route_id=route_id, dir=dir)
		print("Writing", fn)
		with open(fn, 'w') as f :
			f.write(gpx.to_xml())


## ===================== PLAY :


## ================== OPTIONS :

OPTIONS = {
	'SHAPE': write_route_shape,
	# 'STOPS': write_route_stops,
}

## ==================== ENTRY :

if (__name__ == "__main__"):
	assert (commons.parse_options(OPTIONS))
