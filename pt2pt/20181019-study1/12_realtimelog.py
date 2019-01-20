#!/usr/bin/python3

# RA, 2018-10-31

## ================== IMPORTS :

from helpers import commons

import time
import json
import inspect
from collections import defaultdict


## ==================== NOTES :

# Get logged files with
# scp -r w:~/repos/numpde/transport/pt2pt/20181019-study1/OUTPUT/12/Kaohsiung/UV/* ~/*/*/*/*/*study1/OUTPUT/12/Kaohsiung/UV/

pass


## ==================== INPUT :

IFILE = {
	'realtime' : "OUTPUT/12/{city}/UV/{date}/{time}.json",

	'routes' : "OUTPUT/00/ORIGINAL_MOTC/{city}/CityBusApi_Route.json",
}


## =================== OUTPUT :

OFILE = {
	'realtime' : IFILE['realtime'],
}


## ==================== PARAM :

PARAM = {
	'city' : "Taipei",

	'uploads' : [
		{
			'whatisit' : "Kaohsiung_20181105-20181111",
			'basepath' : IFILE['realtime'].format(city="Kaohsiung", date="?", time="?").split('?')[0],
			'repo_url' : "https://osf.io/bwcyv/download",
		},
	]
}


## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))


## ===================== WORK :

def make_log() :
	print("Hello! Run the bash script instead!")


def download() :
	raise NotImplementedError("Retrieving uploads not implemented")


def compress() :
	realtime_files = commons.ls(IFILE['realtime'].format(city=PARAM['city'], date="*", time="*"))
	#print(realtime_files)

	# Allow for pending write operations
	time.sleep(1)

	# Compression 0:
	# zip-base64 contents

	print("COMPRESSION 0")

	# Brutal compression step
	for fn in realtime_files :
		continue
		#commons.zipjson_dump(commons.zipjson_load(fn), fn)


	print("COMPRESSION I: Remove duplicates in back-to-back records")

	for (fn1, fn2) in zip(realtime_files[:-1], realtime_files[1:]) :
		def hashable(J) :
			assert(type(J) is list)
			return list(map(json.dumps, J))

		def unhashable(J) :
			assert(type(J) is list)
			return list(map(json.loads, J))

		try :
			J1 = set(hashable(commons.zipjson_load(fn1)))
			J2 = set(hashable(commons.zipjson_load(fn2)))
		except EOFError :
			# Raised by zipjson_load if a file is empty
			continue
		except Exception as e :
			print("Warning: Cannot open {}/{} ({}).".format(fn1, fn2, e))
			continue

		if not J1.intersection(J2) :
			continue

		J1 = J1.difference(J2)

		J1 = list(unhashable(list(J1)))
		J2 = list(unhashable(list(J2)))

		print("Compressing", fn1)
		commons.zipjson_dump(J1, fn1)


	print("COMPRESSION II: Remove redundancies from individual records")

	# Route meta
	R = commons.zipjson_load(IFILE['routes'].format(city=PARAM['city']))

	# Reindex by subroute-direction
	S = defaultdict(dict)
	for r in R :
		for s in r['SubRoutes'] :
			sid = s['SubRouteUID']
			dir = s['Direction']
			assert(dir not in S[sid])
			S[sid][dir] = s
	#
	S = dict(S)

	# Reindex by RouteUID
	assert(commons.all_distinct([g['RouteUID'] for g in R]))
	R = { g['RouteUID'] : g for g in R }

	def remove_single_route_redundancies(j) :

		subroute_id = j['SubRouteUID']

		if not (subroute_id in S) :
			print("Warning: Unknown subroute {}".format(subroute_id))
			return j

		assert(j['Direction'] in S[subroute_id])
		s = S[subroute_id][j['Direction']]

		for key in ['SubRouteName', 'SubRouteID'] :
			if key in j :
				if not (j[key] == s[key]) :
					print("Warning: Unexpected attribute value {}={}".format(key, j[key]))
				else :
					del j[key]

		if ('RouteUID' in j) :
			route_id = j['RouteUID']
			assert(route_id in R)
			r = R[route_id]

			for key in ['RouteName', 'RouteID'] :
				if key in j :
					if not (j[key] == r[key]) :
						print("Warning: Unexpected attribute value {}={}".format(key, j[key]))
					else :
						del j[key]

			assert(j['RouteUID'] == j['SubRouteUID'])
			del j['RouteUID']

		assert('GPSTime' in j)

		for key in ['SrcUpdateTime', 'UpdateTime']:
			if key in j:
				del j[key]

		# Note:
		#  - we keep the 'OperatorID' field, even if s['OperatorIDs'] has length 1
		#  - of the time stamps, we keep 'GPSTime' which is the bus on-board time

		return j

	for fn in realtime_files :
		try :
			J = commons.zipjson_load(fn)
		except EOFError :
			print("Warning: {} appears empty".format(fn))
			continue
		except Exception :
			print("Warning: Failed to open {}".format(fn))
			continue

		b = len(json.dumps(J)) # Before compression

		try :
			J = list(map(remove_single_route_redundancies, J))
		except ValueError as e :
			print("Warning: ValueError at {} -- {}".format(fn, e))
			continue
		except AssertionError as e :
			print("Warning: Assertion error at {} -- {}".format(fn, e))
			continue
		except Exception as e :
			print("Warning: Compression attempt failed for {} -- {}".format(fn, e))
			continue

		# J = remove_global_route_redundancies(J)
		a = len(json.dumps(J)) # After compression

		assert(a <= b)
		if (a == b) : continue

		print("Compressing", fn)
		commons.zipjson_dump(J, fn)


	print("DONE")


## ================== OPTIONS :

OPTIONS = {
	'MAKE_LOG' : make_log,
	'DOWNLOAD' : download,
	'COMPRESS' : compress,
}


## ==================== ENTRY :

if (__name__ == "__main__") :
	commons.parse_options(OPTIONS)
