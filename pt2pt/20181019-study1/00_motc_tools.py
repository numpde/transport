#!/usr/bin/python3

# RA, 2018-11-06

## ================== IMPORTS :

from helpers import commons
from helpers import maps

import gpxpy, gpxpy.gpx

import os
import re
import json
import math
import time
import inspect
import difflib
import subprocess

# https://stackoverflow.com/questions/29125228/python-matplotlib-save-graph-without-showing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm

## ==================== NOTES :

pass


## ==================== PARAM :

PARAM = {
	'mapbox_api_token' : open(".credentials/UV/mapbox-token.txt", 'r').read(),
	'mapbox_cache_dir' : "helpers/maps_cache/UV/",

	'City' : "Kaohsiung",

	# Need a font for traditional Chinese on images
	'font' : "ORIGINALS/fonts/UV/NotoSerifTC/NotoSerifTC-Light.otf",
}


## ==================== INPUT :

IFILE = {
	'MOTC_routes' : "OUTPUT/00/ORIGINAL_MOTC/{City}/CityBusApi_StopOfRoute.json",
	'MOTC_shapes' : "OUTPUT/00/ORIGINAL_MOTC/{City}/CityBusApi_Shape.json",
	'MOTC_stops'  : "OUTPUT/00/ORIGINAL_MOTC/{City}/CityBusApi_Stop.json",
}

for (k, s) in IFILE.items() : IFILE[k] = s.format(City=PARAM['City'])

## =================== OUTPUT :

OFILE = {
	'Route_GPX' : "OUTPUT/00/GPX/{City}/UV/route_{{route_id}}.gpx",
	'Route_Img' : "OUTPUT/00/img/{City}/UV/route_{{route_id}}.png",
}

for (k, s) in OFILE.items() : OFILE[k] = s.format(City=PARAM['City'])

commons.makedirs(OFILE)

## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))

# Parse a string like
# LINESTRING(lon1 lat1, lon2 lat2, ...)
# into the dictionary
# {'Lon': [lon1, ..], 'Lat': [lat1, ..]}
def parse_linestring(linestring) :
	return dict(zip(
		['Lon', 'Lat'],
		[list(map(float, coo)) for coo in zip(*re.findall(r'(?P<lon>[0-9.]+)[ ](?P<lat>[0-9.]+)', linestring))]
	))


## ============== ASSUMPTIONS :

# Returns a dictionary of routes, indexed by route ID
# with stops and shapes attached
def get_routes() :

	routeid_of = (lambda r: r['RouteUID'])

	# I. Get the list of route descriptions, including the stops
	motc_routes = commons.zipjson_load(IFILE['MOTC_routes'])

	try :
		# As of 2018-11-09, the following passes for KHH but not TPE
		assert(all((route['RouteUID'] == route['SubRouteUID']) for route in motc_routes))
	except:
		pass

	try:
		# Are route IDs distinct? No ...
		assert(commons.all_distinct(map(routeid_of, motc_routes)))
	except AssertionError :
		# ... because direction = 0 and 1 appear separately
		pass

	# Reindex routes by route ID
	motc_routes = commons.index_dicts_by_key(motc_routes, routeid_of, keys_singletons_ok=['Direction', 'Stops'])

	try :
		# As of 2018-11-11, there may be up to 12 subroutes in a route in Taipei
		assert(12 >= max(len(route['Stops']) for route in motc_routes.values()))
	except :
		pass

	# II. Now attach "shapes" to routes

	# List of route "shapes"
	motc_shapes = commons.zipjson_load(IFILE['MOTC_shapes'])

	# The shapes do not contain certain ID fields
	assert(all(set(shape.keys()).isdisjoint({'SubRouteUID', 'SubRouteID'}) for shape in motc_shapes))

	# Parse the Geometry Linestring of shapes
	for (n, _) in enumerate(motc_shapes) :
		motc_shapes[n]['Geometry'] = parse_linestring(motc_shapes[n]['Geometry'])

	# This will be the ID field
	assert(all(routeid_of(shape) for shape in motc_shapes))

	try :
		# This passes for TPE but not for KHH
		assert(commons.all_distinct(routeid_of(shape) for shape in motc_shapes))
	except :
		pass

	# Index shapes by route ID
	motc_shapes = commons.index_dicts_by_key(motc_shapes, routeid_of, keys_singletons_ok=['Direction', 'Geometry'])

	try :
		# As of 2018-11-11:
		# For KHH, generally have a shape for each SubRouteUID
		# For TPE, there is only one shape per RouteUID

		# This does not hold for TPE
		assert(all(('Direction' in shape) for shape in motc_shapes))
	except :
		# Append a dummy Direction field to shapes
		for (j, shape) in motc_shapes.items() :
			if not ('Direction' in shape) :
				motc_shapes[j]['Direction'] = [None] * len(shape['Geometry'])
				# Actually, for TPE, we have:
				assert(1 == len(shape['Geometry']))

	for (i, r) in motc_routes.items() :

		# No shape for this route?
		if not (i in motc_shapes.keys()) :
			motc_routes[i]['Shape'] = []
			continue

		# True for KHH an TPE
		assert(commons.all_distinct(motc_shapes[i]['Direction']))
		assert(len(motc_shapes[i]['Direction']) <= 2)

		# Attach a list of shapes with Direction tag (possibly None)
		# Note: 'None' is preserved by JSON (https://stackoverflow.com/a/3548740/3609568)
		motc_routes[i]['Shape'] = [
			{
				'Direction' : dir,
				'Geometry'  : geo,
			}
			for (dir, geo) in zip(motc_shapes[i]['Direction'], motc_shapes[i]['Geometry'])
		]

	# Show all info of a route
	# print(next(iter(motc_routes.values())))

	return motc_routes

## ===================== WORK :

def motc_download() :
	print("Use the bash script for downloading MOTC data")

def write_route_img() :

	if not os.path.isfile(PARAM['font']) :
		print("Warning: Font not found, text may be unreadable")
		time.sleep(2)

	for (route_id, route) in get_routes().items() :

		# Get the Chinese and English route name (unless they are the same)
		names = list(map(str.strip, commons.inspect({'RouteName': ['Zh_tw', 'En']})(route)))
		if not commons.all_distinct(names) : names = set(names)
		route_name = " / ".join(names)

		print("Route {}: {}".format(route_id, route_name))

		outfile = OFILE['Route_Img'].format(route_id=route_id)

		# Skip if the image already exists
		#if os.path.isfile(outfile) : continue

		route_dirs = route['Direction']

		(fig, ax) = plt.subplots()

		# Colors from the default pyplot palette
		C = [ ("C{}".format(n % 10)) for n in range(len(route_dirs)) ]

		for (shape, c) in zip(route['Shape'], C) :
			(y, x) = commons.inspect({'Geometry': ('Lat', 'Lon')})(shape)
			ax.plot(x, y, '--', c=(c if shape['Direction'] else 'k'), linewidth=1, zorder=0)

		for (stops, c) in zip(route['Stops'], C) :
			(y, x) = zip(*map(commons.inspect({'StopPosition' : ('PositionLat', 'PositionLon')}), stops))
			ax.scatter(x, y, c=c, s=10, zorder=100)

		# Get the dimensions of the plot
		(left, right, bottom, top) = ax.axis()

		# Compute a nicer aspect ratio if it is too narrow
		(w, h, phi) = (right - left, top - bottom, (1 + math.sqrt(5)) / 2)
		if (w < h / phi) : (left, right) = (((left + right) / 2 + s * h / phi / 2) for s in (-1, +1))
		if (h < w / phi) : (bottom, top) = (((bottom + top) / 2 + s * w / phi / 2) for s in (-1, +1))

		# Set new dimensions
		ax.axis([left, right, bottom, top])

		# Label plot with the route name
		ax.text(
			0.5, 0.95,
			route_name,
			ha='center', va='top',
			wrap=True,
			transform=ax.transAxes,
			zorder=1000,
			color='black',
			#fontname="Noto Sans CJK",
			fontproperties=mfm.FontProperties(fname=PARAM['font']),
			fontsize='x-small',
	        bbox=dict(boxstyle="square", ec=(0.5, 0.5, 1), fc=(0.8, 0.8, 1), alpha=0.7),
        )

		# Get the dimensions of the plot (again)
		(left, right, bottom, top) = ax.axis()
		# Bounding box for the map
		bbox = (left, bottom, right, top)

		# Remove labels
		ax.axis('off')
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# Download the background map
		i = maps.get_map_by_bbox(bbox, token=PARAM['mapbox_api_token'], cachedir=PARAM['mapbox_cache_dir'])

		# Apply the background map
		img = ax.imshow(i, extent=(left, right, bottom, top), interpolation='quadric', zorder=-100)

		# Save image to disk
		fig.savefig(
			outfile,
			bbox_inches='tight', pad_inches=0,
			dpi=180
		)

		plt.pause(0.1)
		plt.close(fig)

		time.sleep(0.2)



def write_route_gpx() :

	for (route_id, route) in get_routes().items() :

		# Object to organize the GPS tracks and Waypoints
		gpx = gpxpy.gpx.GPX()
		gpx.name = "Route: {} / {}".format(*commons.inspect({'RouteName': ['Zh_tw', 'En']})(route))
		gpx.description  = "Route ID: {}".format(route_id)

		outfile = OFILE['Route_GPX'].format(route_id=route_id)

		open = commons.logged_open

		# Create a track in our GPX
		gpx_track = gpxpy.gpx.GPXTrack()
		gpx.tracks.append(gpx_track)

		for shape in route['Shape'] :
			# Create a segment in our GPX track
			gpx_segment = gpxpy.gpx.GPXTrackSegment()

			# Insert GPS points
			for (p, q) in zip(*commons.inspect({'Geometry': ['Lat', 'Lon']})(shape)):
				gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=p, longitude=q))

			gpx_track.segments.append(gpx_segment)

		# If there are no distinct subroutes, make a dummy list
		subroute_id = route['SubRouteUID']
		if type(subroute_id) is str :
			assert(subroute_id == route_id)
			route['SubRouteUID'] = [None] * len(route['Stops'])

		for (subroute_id, dir, stops) in zip(*commons.inspect(['SubRouteUID', 'Direction', 'Stops'])(route)) :
			for stop in stops :
				(p, q) = commons.inspect({'StopPosition': ('PositionLat', 'PositionLon')})(stop)
				stop_name = "{}/{} #{}: {} / {}".format(subroute_id or "", dir, stop['StopSequence'], stop['StopName'].get('Zh_tw', "--"), stop['StopName'].get('En', "--"))
				stop_desc = "{} ({})".format(stop['StopUID'], stop['StationID'])
				wp = gpxpy.gpx.GPXWaypoint(latitude=p, longitude=q, name=stop_name, description=stop_desc)
				gpx.waypoints.append(wp)

		with open(outfile, 'w') as f :
			f.write(gpx.to_xml())


# Look for bus stops or bus routes based on user input
def interactive_search() :

	motc_stops = commons.index_dicts_by_key(commons.zipjson_load(IFILE['MOTC_stops']), (lambda r: r['StopUID']))
	motc_routes = commons.index_dicts_by_key(commons.zipjson_load(IFILE['MOTC_routes']), (lambda r: r['SubRouteUID']),
											 keys_singletons_ok=['Direction', 'Stops'])

	# Only keep essential info about a stop or a route
	def slim(D):
		return {
			k : D[k]
			for k in ['StopUID', 'StopName'] + ['SubRouteUID', 'SubRouteName']
			if k in D
		}

	def strip_brackets(s):
		return re.match(r'(?P<name>[^(]+)[ ]*(?P<extra>\(\w+\))*', s).group('name').strip()

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

		command = input("Enter command ([s]top, [S]top, [r]oute, [R]oute): ").strip()[0:1]
		if not command : break
		if not command in ['s', 'S', 'r', 'R'] : continue

		query = input("Enter search string: ").strip()
		if not query : continue

		# Collect search results here
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
				result.append(commons.pretty_json(stop))

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

				result.append(commons.pretty_json(route))

				for (dir, stops) in zip(route['Direction'], stops) :
					result.append("Route-Direction {}-{}:".format(route_id, dir))
					result.append(commons.pretty_json(stops))

		try :
			subprocess.run(["less"], input='\n'.join(result).encode('utf-8'))
		except :
			print(*result, sep='\n')
			print("(Could not open 'less' as subprocess)")

		continue


## ================== OPTIONS :

OPTIONS = {
	'DOWNLOAD'    : motc_download,
	'ROUTE_GPX'   : write_route_gpx,
	'ROUTE_IMG'   : write_route_img,
	'SEARCH'      : interactive_search,
}


## ==================== ENTRY :

if (__name__ == "__main__"):
	commons.parse_options(OPTIONS)
