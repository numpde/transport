#!/usr/bin/python3

# RA, 2018-11-06

## ================== IMPORTS :

import commons

import gpxpy, gpxpy.gpx
import re
import json
import inspect


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
	'Route_GPX' : "OUTPUT/00/GPX/Kaohsiung/UV/route_{route_id}_{dir}.gpx",
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

def write_route_gpx() :

	# Note: index by RouteUID (not SubRouteUID) because...
	motc_routes = commons.index_dicts_by_key(commons.zipjson_load(IFILE['MOTC_routes']), (lambda r: r['RouteUID']), preserve_singletons=['Direction', 'Stops'])
	# ...this file only provides a RouteUID for each record
	motc_shapes = commons.zipjson_load(IFILE['MOTC_shapes'])

	for shape in motc_shapes :

		(route_id, dir) = (shape['RouteUID'], shape['Direction'])

		route = motc_routes[route_id]

		# Parse LINESTRING
		(lon, lat) = tuple(
			list(map(float, coo))
			for coo in zip(*re.findall(r'(?P<lon>[0-9.]+)[ ](?P<lat>[0-9.]+)', shape['Geometry']))
		)

		gpx = gpxpy.gpx.GPX()

		if dir in route['Direction'] :
			for stop in dict(zip(route['Direction'], route['Stops']))[dir] :
				(p, q) = (stop['StopPosition']['PositionLat'], stop['StopPosition']['PositionLon'])
				stop_name = "{}-#{}: {} / {}".format(dir, stop['StopSequence'], stop['StopName']['Zh_tw'], stop['StopName']['En'])
				stop_desc = "{} ({})".format(stop['StopUID'], stop['StationID'])
				wp = gpxpy.gpx.GPXWaypoint(latitude=p, longitude=q, name=stop_name, description=stop_desc, )
				gpx.waypoints.append(wp)
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

		fn = OFILE['Route_GPX'].format(route_id=route_id, dir=dir)
		print("Writing", fn)
		with open(fn, 'w') as f :
			f.write(gpx.to_xml())


## ===================== PLAY :


## ================== OPTIONS :

OPTIONS = {
	'ROUTE_GPX': write_route_gpx,
	# 'STOPS': write_route_stops,
}


## ==================== ENTRY :

if (__name__ == "__main__"):
	assert(commons.parse_options(OPTIONS))
