#!/usr/bin/python3

# RA, 2018-10-31

## ================== IMPORTS :

import os
import sys
import json
import glob
import inspect
from collections import defaultdict

## ==================== NOTES :

pass


## ==================== INPUT :

IFILE = {
	'response' : "OUTPUT/12/Kaohsiung/UV/{dt}.json",

	'routes' : "ORIGINALS/MOTC/Kaohsiung/CityBusApi_Route/data.json",
}


## =================== OUTPUT :

OFILE = {
	'response' : IFILE['response'],
}

# Create output directories
#for f in OFILE.values() : if f : os.makedirs(os.path.dirname(f), exist_ok=True)


## ==================== PARAM :

PARAM = {
}

## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))

# Log which files are opened
def logged_open(filename, mode='r', *argv, **kwargs) :
	print("({}):\t{}".format(mode, filename))
	return open(filename, mode, *argv, **kwargs)

def is_distinct(L) :
	return (len(set(L)) == len(list(L)))

## ===================== WORK :

def download() :

	print("Hello! Run the bash script instead!")


def compress() :

	response_files = sorted(glob.glob(IFILE['response'].format(dt="*")))
	#print(response_files)

	# Compression I:
	# Remove records from file if present in the subsequent file

	print("COMPRESSION I")

	for (fn1, fn2) in zip(response_files[:-1], response_files[1:]) :
		def hashable(J) :
			assert(type(J) is list)
			return list(map(json.dumps, J))

		def unhashable(J) :
			assert(type(J) is list)
			return list(map(json.loads, J))

		J1 = set(hashable(json.load(open(fn1, 'r'))))
		J2 = set(hashable(json.load(open(fn2, 'r'))))

		if (len(J1.intersection(J2)) == 0) :
			continue

		J1 = J1.difference(J2)

		J1 = list(unhashable(list(J1)))
		J2 = list(unhashable(list(J2)))

		json.dump(J1, open(fn1, 'w'))


	# Compression II:
	# Remove route names if available elsewhere

	print("COMPRESSION II")

	# Route meta
	R = json.load(open(IFILE['routes'], 'r'))

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
	assert(is_distinct([g['RouteUID'] for g in R]))
	R = { g['RouteUID'] : g for g in R }

	def remove_route_redundancies(j) :

		subroute_id = j['SubRouteUID']
		assert(subroute_id in S)
		assert(j['Direction'] in S[subroute_id])
		s = S[subroute_id][j['Direction']]

		for key in ['SubRouteName', 'SubRouteID'] :
			if (key in j) :
				assert(j[key] == s[key])
				del j[key]

		if ('RouteUID' in j) :
			route_id = j['RouteUID']
			assert(route_id in R)
			r = R[route_id]

			for key in ['RouteName', 'RouteID'] :
				if (key in j) :
					assert(j[key] == r[key])
					del j[key]

			assert(j['RouteUID'] == j['SubRouteUID'])
			del j['RouteUID']

		# Note:
		#  - we keep the 'OperatorID' field, even if s['OperatorIDs'] has length 1

		return j

	for fn in response_files :
		J = json.load(open(fn, 'r'))
		b = len(json.dumps(J))
		J = list(map(remove_route_redundancies, J))
		a = len(json.dumps(J))

		assert(a <= b)
		if (a == b) : continue

		print("Compressing", fn)
		json.dump(J, open(fn, 'w'))


## ================== OPTIONS :

OPTIONS = {
	'DOWNLOAD' : download,
	'COMPRESS' : compress,
}

def parse_options() :

	if (len(sys.argv) > 1):
		OPT = sys.argv[1]
		ARG = sys.argv[2:]
		for (opt, fun) in OPTIONS.items():
			if (opt == OPT):
				fun(*ARG)
				return True

		raise RuntimeError("Unrecognized command line option")

	return False


## ==================== ENTRY :

if (__name__ == "__main__") :

	if not parse_options() :

		print("Please specify option via command line:", *OPTIONS.keys())

