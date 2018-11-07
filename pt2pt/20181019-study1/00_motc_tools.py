#!/usr/bin/python3

# RA, 2018-11-06

## ================== IMPORTS :

from helpers import commons

import gpxpy, gpxpy.gpx
import re
import json
import inspect
import difflib
import subprocess


## ==================== NOTES :

pass


## ==================== INPUT :

IFILE = {
	'MOTC_routes' : "OUTPUT/00/ORIGINAL_MOTC/Kaohsiung/CityBusApi_StopOfRoute/data.json",
	'MOTC_shapes' : "OUTPUT/00/ORIGINAL_MOTC/Kaohsiung/CityBusApi_Shape/data.json",
	'MOTC_stops'  : "OUTPUT/00/ORIGINAL_MOTC/Kaohsiung/CityBusApi_Stop/data.json",
}


## =================== OUTPUT :

OFILE = {
	'Route_GPX' : "OUTPUT/00/GPX/Kaohsiung/UV/route_{route_id}-{dir}.gpx",
}

commons.makedirs(OFILE)


## ==================== PARAM :

PARAM = {
}


## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))

# For printing a JSON nicely
def pretty(J):
	return json.dumps(J, indent=2, ensure_ascii=False)


## ===================== WORK :

def motc_download() :
	print("Use the bash script for downloading MOTC data")

def write_route_gpx() :

	# Note: index by RouteUID (not SubRouteUID) because...
	motc_routes = commons.index_dicts_by_key(commons.zipjson_load(IFILE['MOTC_routes']), (lambda r: r['RouteUID']), preserve_singletons=['Direction', 'Stops'])
	# ...this file only provides a RouteUID for each record
	motc_shapes = commons.zipjson_load(IFILE['MOTC_shapes'])

	open = commons.logged_open

	issues = []

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
				(p, q) = commons.inspect({'StopPosition': ('PositionLat', 'PositionLon')})(stop)
				stop_name = "{}-#{}: {} / {}".format(dir, stop['StopSequence'], stop['StopName']['Zh_tw'], stop['StopName']['En'])
				stop_desc = "{} ({})".format(stop['StopUID'], stop['StationID'])
				wp = gpxpy.gpx.GPXWaypoint(latitude=p, longitude=q, name=stop_name, description=stop_desc)
				gpx.waypoints.append(wp)
		else :
			issues.append("Route {}, direction {} not found in MOTC_routes".format(route_id, dir))

		# Create first track in our GPX
		gpx_track = gpxpy.gpx.GPXTrack()
		gpx.tracks.append(gpx_track)

		# Create first segment in our GPX track
		gpx_segment = gpxpy.gpx.GPXTrackSegment()
		gpx_track.segments.append(gpx_segment)

		# Create points
		for (p, q) in zip(lat, lon) :
			gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=p, longitude=q))

		with open(OFILE['Route_GPX'].format(route_id=route_id, dir=dir), 'w') as f :
			f.write(gpx.to_xml())

		continue

	print("Issues:")
	for issue in (issues or ["None"]) : print(issue)


# Look for bus stops or bus routes based on user input
def interactive_search() :

	motc_stops = commons.index_dicts_by_key(commons.zipjson_load(IFILE['MOTC_stops']), (lambda r: r['StopUID']))
	motc_routes = commons.index_dicts_by_key(commons.zipjson_load(IFILE['MOTC_routes']), (lambda r: r['SubRouteUID']), preserve_singletons=['Direction', 'Stops'])

	# Only keep essential info about a stop or a route
	def slim(D):
		return {
			k : D[k]
			for k in ['StopUID', 'StopName'] + ['SubRouteUID', 'SubRouteName']
			if k in D
		}

	def strip_brackets(s):
		return re.match(r'(?P<name>[^\(]+)[ ]*(?P<extra>\(\w+\))*', s).group('name').strip()

	def matchratio_names(name1, name2):
		return difflib.SequenceMatcher(None, name1, name2).ratio()

	def matchratio(query, keywords) :
		return max([matchratio_names(query.lower(), keyword.lower()) for keyword in keywords])

	# Search keywords for bus stops
	motc_s_kw = {
		j : [
				stop[k]
				for k in ['StopUID', 'StopID', 'StationID']
			] + [
				str(coo)
				for coo in stop['StopPosition'].values()
			] + [
				strip_brackets(name)
				for name in stop['StopName'].values()
			]
		for (j, stop) in motc_stops.items()
	}

	# Search keywords for bus routes
	motc_r_kw = {
		j : [
				route[k]
				for k in ['RouteUID', 'RouteID', 'SubRouteUID', 'SubRouteID']
			] + [
				strip_brackets(name)
				for name in route['SubRouteName'].values()
			]
		for (j, route) in motc_routes.items()
	}

	# Interaction
	while True :

		q = input("Enter command (s, S, r, R):\n").strip().split(' ')

		# No input
		if not q[0] : break

		# No arguments
		if (len(q) < 2) : continue

		(command, query) = (q[0], " ".join(q[1:]))

		result = []

		# Search for bus stops
		if (command.lower() == "s") :

			top_match_motc = sorted(
				motc_stops.items(),
				key=(lambda item : matchratio(query, motc_s_kw[item[0]])),
				reverse=True
			)[0:25]

			if (command == 's') :

				result.append("Suggestions:")
				for (_, stop) in top_match_motc : result.append(slim(stop))

			else :

				(_, stop) = top_match_motc[0]
				result.append(pretty(stop))

		# Search for bus routes
		if (command.lower() == "r"):

			top_match_motc = sorted(
				motc_routes.items(),
				key=(lambda item : matchratio(query, motc_r_kw[item[0]])),
				reverse=True
			)[0:10]

			if (command == 'r') :

				result.append("Suggestions:")
				for (_, route) in top_match_motc : result.append(slim(route))

			else :

				(route_id, route) = top_match_motc[0]

				stops = route['Stops']
				route['Stops'] = '[see below]'

				result.append(pretty(route))

				for (dir, stops) in zip(route['Direction'], stops) :
					result.append("Route-Direction {}-{}:".format(route_id, dir))
					result.append(pretty(stops))

		try :
			subprocess.run(["less"], input='\n'.join(result).encode('utf-8'))
		except :
			print(*result, sep='\n')
			print("(Could not open 'less' as subprocess)")

		continue


## ===================== PLAY :


## ================== OPTIONS :

OPTIONS = {
	'DOWNLOAD'  : motc_download,
	'ROUTE_GPX' : write_route_gpx,
	'SEARCH'    : interactive_search,
}


## ==================== ENTRY :

if (__name__ == "__main__"):
	commons.parse_options(OPTIONS)
