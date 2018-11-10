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

	'City' : "Taipei",

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
	'Route_GPX' : "OUTPUT/00/GPX/{City}/UV/route_{{route_id}}-{{dir}}.gpx",
	'Route_Img' : "OUTPUT/00/img/{City}/UV/route_{{route_id}}.png",
}

for (k, s) in OFILE.items() : OFILE[k] = s.format(City=PARAM['City'])

commons.makedirs(OFILE)

## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))

# For printing a JSON nicely
def pretty(J):
	return json.dumps(J, indent=2, ensure_ascii=False)

# Parse a string like
# LINESTRING(lon1 lat1, lon2 lat2, ...)
# into ([lon1, ..], [lat1, ..])
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
		# As of 2018-11-09, the following passes for Kaohsiung but not Taipei
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
	motc_routes = commons.index_dicts_by_key(
		motc_routes,
		routeid_of,
		preserve_singletons=['SubRouteUID', 'Direction', 'Stops']
	)

	try :
		# As of 2018-11-11, there may be up to 12 subroutes in a route in Taipei
		assert(12 >= max(len(route['SubRouteUID']) for route in motc_routes.values()))
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

	# This passes for TPE (and KHH?)
	assert(commons.all_distinct(routeid_of(shape) for shape in motc_shapes))

	# Index shapes by route ID
	motc_shapes = commons.index_dicts_by_key(
		motc_shapes,
		routeid_of,
		preserve_singletons=['Direction', 'Geometry']
	)

	try :
		# As of 2018-11-11:
		# In Kaohsiung, the shapes mostly correspond to Directions
		# In Taipei, there is one shape per RouteID

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

		# Directions of this route
		route_dirs = r['Direction']

		# No shape for this route?
		if not (i in motc_shapes.keys()) :
			#print("No shape for route {}.".format(i))
			continue

		# True for KHH an TPE
		assert(commons.all_distinct(motc_shapes[i]['Direction']))
		assert(len(motc_shapes[i]['Direction']) <= 2)

		# Attach a list of shapes with Direction tag (possibly None)
		# Note: 'None' is preserved in JSON here (https://stackoverflow.com/a/3548740/3609568)
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
			dir = shape['Direction']
			(y, x) = commons.inspect({'Geometry': ('Lat', 'Lon')})(shape)
			ax.plot(x, y, '--', c=c, zorder=0)

		for (stops, c) in zip(route['Stops'], C) :
			(y, x) = zip(*map(commons.inspect({'StopPosition' : ('PositionLat', 'PositionLon')}), stops))
			ax.scatter(x, y, c=c, zorder=100)

		# Get the dimensions of the plot
		(left, right, bottom, top) = ax.axis()

		# Compute a nice aspect ratio
		(w, h, phi) = (right - left, top - bottom, (1 + math.sqrt(5)) / 2)
		if (w < h / phi) :
			(left, right) = (((left + right) / 2 + (s * h / phi) / 2) for s in (-1, +1))
		if (h < w / phi) :
			(bottom, top) = (((bottom + top) / 2 + (s * w / phi) / 2) for s in (-1, +1))

		# Set new dimensions
		ax.axis([left, right, bottom, top])

		# Label plot with the route name
		ax.text(
			0.5, 0.95,
			route_name,
			wrap=True,
			transform=ax.transAxes,
			zorder=1000,
			color='black',
			#fontname="Noto Sans CJK",
			fontproperties=mfm.FontProperties(fname=PARAM['font']),
			fontsize='x-small',
			ha='center', va='top',
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
		i = maps.get_map_by_bbox(bbox, token=PARAM['mapbox_api_token'])

		# Apply the background map
		img = ax.imshow(i, extent=(left, right, bottom, top), interpolation='quadric', zorder=-100)

		# Save image to disk
		fig.savefig(
			outfile,
			bbox_inches='tight', pad_inches=0,
			dpi=180
		)

		plt.pause(0.1)
		#plt.show()
		plt.close(fig)

		time.sleep(0.2)



def write_route_gpx() :

	# TODO: replace by
	# motc_routes = get_routes()

	# Note: index by RouteUID (not SubRouteUID) because...
	motc_routes = commons.index_dicts_by_key(commons.zipjson_load(IFILE['MOTC_routes']), (lambda r: r['RouteUID']), preserve_singletons=['Direction', 'Stops'])
	# ...this file only provides a RouteUID for each record
	motc_shapes = commons.zipjson_load(IFILE['MOTC_shapes'])

	open = commons.logged_open

	issues = []

	for shape in motc_shapes :

		(route_id, dir) = (shape['RouteUID'], shape['Direction'])

		route = motc_routes[route_id]

		# Parse the route "shape"
		(lon, lat) = commons.inspect(('Lon', 'Lat'))(parse_linestring(shape['Geometry']))

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
	'DOWNLOAD'    : motc_download,
	'ROUTE_GPX'   : write_route_gpx,
	'ROUTE_IMG'   : write_route_img,
	'SEARCH'      : interactive_search,
}


## ==================== ENTRY :

if (__name__ == "__main__"):
	commons.parse_options(OPTIONS)
