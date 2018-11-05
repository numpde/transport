#!/usr/bin/python3

# RA, 2018-11-03

## ================== IMPORTS :

import commons

import re
import difflib
import pickle
import json
import glob
import time
import inspect
from itertools import chain
from collections import defaultdict
#import matplotlib.pyplot as plt


## ==================== NOTES :

pass


## ==================== INPUT :

IFILE = {
	'OSM'  : "OUTPUT/02/UV/kaohsiung.pkl",
	'MOTC_routes' : "ORIGINALS/MOTC/Kaohsiung/CityBusApi_StopOfRoute/data.json",
	'MOTC_stops' : "ORIGINALS/MOTC/Kaohsiung/CityBusApi_Stop/data.json",
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

# Print a JSON nicely
def pretty_print(J) :
	print(json.dumps(J, indent=2, ensure_ascii=False))

## ===================== WORK :

def check_osm_stops() :

	print("Loading OSM...")

	OSM = pickle.load(open(IFILE['OSM'], 'rb'))

	print("Extracting stops...")

	stops = {
		n : (OSM['node_tags'].get(n), route)
		for (route_id, route) in OSM['rels']['route'].items()
		if (route['t'].get('route') == 'bus')
		for n in route['n']
	}

	del OSM

	print("Launching check")
	time.sleep(1)

	for (n, (stop, route)) in stops.items() :

		print("")
		print(" === Checking node #{}".format(n))
		print("")

		if not stop :
			print("ISSUE: Stop not found")
			print("Stop:", stop)
			print("Route:", route)
			input("Press ENTER")
			continue

		#print(n)

		if not (stop.get('public_transport') == 'platform') :
			print("ISSUE: Stop missing proper tag public_transport=platform")
			print("Stop:", stop)
			print("Route:", route)
			input("Press ENTER")


		if not (stop.get('bus') == 'yes') :
			print("ISSUE: Stop should have a bus=yes tag")
			print("Stop:", stop)
			print("Route:", route)
			input("Press ENTER")

		if not stop.get("name") :
			print("ISSUE: Stop has no name")
			print("Stop:", stop)
			print("Route:", route)
			input("Press ENTER")

	return

# Check compliance with the guidelines
# https://wiki.openstreetmap.org/wiki/Tag:route%3Dbus
# https://wiki.openstreetmap.org/wiki/Buses
def check_osm_routes() :

	OSM = pickle.load(open(IFILE['OSM'], 'rb'))

	for (route_id, route) in OSM['rels']['route'].items():

		# Dictionary of OSM tags
		route_tags = route['t']

		# Skip non-bus routes
		if not (route_tags.get('route') == 'bus'): continue

		# 'name' is not mandatory but seems to be always present
		route_name = route_tags.get('name', "NO_NAME")

		# Expect 'type' to be either route or route_master
		assert(route_tags.get('type')), "(mandatory)"
		assert(route_tags['type'] in ['route', 'route_master'])

		print("")
		print("Checking route #{} '{}':".format(route_id, route_name))
		print("")

		if not (route_tags.get('type') == 'route') :
			print("ISSUE: tag type=route not found")
			print(route)
			input("Press ENTER")
      
		if not (route_tags.get('route') == 'bus') :
			print("ISSUE: tag route=bus not found")
			print(route)
			input("Press ENTER")

		if not route_tags.get('ref') :
			print("ISSUE: tag 'ref' not found")
			print(route)
			input("Press ENTER")

		if not (route_tags.get('roundtrip') in ['yes', 'no']) :
			print("ISSUE: tag roundtrip=yes/no not found")
			print(route)
			input("Press ENTER")
	
		if not (route_tags.get('public_transport:version') == '2') :
			print("ISSUE: tag public_transport:version=2 not found")
			print(route)
			input("Press ENTER")

		if not route_tags.get('name') :
			print("ISSUE: tag 'name' not found")
			print(route)
			input("Press ENTER")


# Look for bus stops or bus routes base on user input
def interactive_match() :

	motc_stops = commons.index_dicts_by_key(commons.zipjson_load(IFILE['MOTC_stops']), (lambda r: r['StopUID']))
	motc_routes = commons.index_dicts_by_key(commons.zipjson_load(IFILE['MOTC_routes']), (lambda r: r['SubRouteUID']))

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

	q = input("\nEnter command (s, S, r, R):\n").strip()
	if not q : return

	q = q.split(' ')

	if (len(q) < 2) :
		interactive_match()
		return

	(query_type, query) = (q[0], " ".join(q[1:]))

	def slim_stop(stop) :
		return {
			k : stop[k]
			for k in ['StopUID', 'StopName']
		}

	def slim_route(route) :
		return {
			k : route[k]
			for k in ['SubRouteUID', 'SubRouteName']
		}


	# Search for bus stops
	if (query_type.lower() == "s") :

		top_match_motc = sorted(
			motc_stops.items(),
			key=(lambda item : matchratio(query, motc_s_kw[item[0]])),
			reverse=True
		)[0:25]

		if (query_type == 's') :

			print("Suggestions:")

			for (_, stop) in top_match_motc :
				print(slim_stop(stop))

		else :

			(_, stop) = top_match_motc[0]
			pretty_print(stop)

	# Search for bus routes
	if (query_type.lower() == "r"):

		top_match_motc = sorted(
			motc_routes.items(),
			key=(lambda item : matchratio(query, motc_r_kw[item[0]])),
			reverse=True
		)[0:10]

		if (query_type == 'r') :

			print("Suggestions:")

			for (route_id, route) in top_match_motc :
				print(slim_route(route))

		else :

			(route_id, route) = top_match_motc[0]

			stops = route['Stops']
			del route['Stops']

			pretty_print(route)

			def zip_listify(a, b):
				return zip(a, b) if (type(a) is list) else zip([a], [b])

			for (dir, stops) in zip_listify(route['Direction'], stops) :
				print("Direction", dir)
				pretty_print(stops)

	interactive_match()
	return

## ===================== PLAY :

def match_routes() :

	# MOTC route info
	motc_routes = commons.index_dicts_by_key(commons.zipjson_load(IFILE['MOTC_routes']), (lambda r: r['SubRouteUID']))

	# for (route_id, route) in route_stops.items() :
	# 	stops = dict(zip(route['Direction'], route['Stops']))

	OSM = pickle.load(open(IFILE['OSM'], 'rb'))

	for (route_id, route) in OSM['rels']['route'].items():

		# Skip non-bus routes
		if not (route['t'].get('route') == 'bus'): continue

		# Note: most routes have relations in route['r']

		(route_tags, route_stops, route_ways) = (route['t'], route['n'], route['w'])

		# https://wiki.openstreetmap.org/wiki/Buses
		route_name = route_tags['name']

		# Common routines

		def strip_brackets(s):
			return re.match(r'(?P<name>\w+)+[ ]*(?P<extra>\(\w+\))*', s).group('name')

		def matchratio_stop_names(name1, name2):
			return difflib.SequenceMatcher(None, strip_brackets(name1), strip_brackets(name2)).ratio()

		# Method 0: Match route names

		top_namematch_motc_ids = None

		try :

			top_namematch_motc_ids = sorted(
				motc_routes.keys(),
				key=(lambda j : matchratio_stop_names(route_name, motc_routes[j]['RouteName']['Zh_tw'])),
				reverse=True
			)[0:6]

			#print("Route {} best matches: {}".format(route_name, ",".join([motc_routes[j]['RouteName']['Zh_tw'] for j in top_namematch_motc_ids])))
		except :
			raise


		# Method 1: Match route start/end stops

		def zip_listify(a, b) :
			return zip(a, b) if (type(a) is list) else zip([a], [b])

		try :
			(route_a, route_b) = (route_tags['from'], route_tags['to'])

			def matchratio_ab(motc_route) :
				# motc_name = motc_route['RouteName']['Zh_tw']
				for (dir, stops) in zip_listify(motc_route['Direction'], motc_route['Stops']) :

					(motc_a, motc_b) = (stops[0]['StopName']['Zh_tw'], stops[-1]['StopName']['Zh_tw'])

					ab_ratio = (matchratio_stop_names(route_a, motc_a) + matchratio_stop_names(route_b, motc_b)) / 2
					assert((0 <= ab_ratio) and (ab_ratio <= 1))

					yield (ab_ratio, { 'SubRouteUID' : motc_route['SubRouteUID'], 'Direction' : dir })

			ab_matchratios = sorted(
				chain.from_iterable([
					matchratio_ab(motc_routes[j]) for j in top_namematch_motc_ids
				]),
				key=(lambda p: p[0]), reverse=True
			)

			print(route_name, ab_matchratios)

		except KeyError as e :
			#print("Method 1 failed on route {}".format(route_name))
			continue

		#print(route_tags)
		continue


		if (len(route_stops) < 2) :
			#print("Route {} has fewer than two stops".format(route_name))
			#print(route_ways)
			continue

		# Method 2: Match all stops

		# Get stop info
		if not all(OSM['node_tags'].get(i) for i in route_stops) :
			print("Nodes of route {} not found".format(route_tags['name']))
			continue

		route_stops = {
			i : OSM['node_tags'].get(i)
			for i in route_stops
		}

		print(route_stops)


		#print(route['n'])
		#time.sleep(1)

	#
	# 	route_name = route['t'].get('name')
	#
	# 	route_ref = route['t']['ref']
	# 	#if (route_ref == '88') :
	# 	print(route_name, route_id, route['t'])
	# exit(39)


	return

## ================== OPTIONS :

OPTIONS = {
	# 'CHECK_STOPS' : check_osm_stops,
	# 'CHECK_ROUTES' : check_osm_routes,
	# 'MATCH' : match_routes,
	'INTERACTIVE' : interactive_match,
}

## ==================== ENTRY :

if (__name__ == "__main__") :

	assert(commons.parse_options(OPTIONS))
