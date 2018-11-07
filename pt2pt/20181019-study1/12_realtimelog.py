#!/usr/bin/python3

# RA, 2018-10-31

## ================== IMPORTS :

from helpers import commons

import time
import json
import glob
import inspect
from collections import defaultdict


## ==================== NOTES :

# Get logged files with
# scp -r w:~/repos/numpde/transport/pt2pt/20181019-study1/OUTPUT/12/Kaohsiung/UV/* ~/*/*/*/*/*study1/OUTPUT/12/Kaohsiung/UV/

pass


## ==================== INPUT :

IFILE = {
	'response' : "OUTPUT/12/Kaohsiung/UV/{d}/{t}.json",

	'routes' : "OUTPUT/00/ORIGINAL_MOTC/Kaohsiung/CityBusApi_Route/data.json",
}


## =================== OUTPUT :

OFILE = {
	'response' : IFILE['response'],
}


## ==================== PARAM :

PARAM = {
}

## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))


## ===================== WORK :

def download() :

	print("Hello! Run the bash script instead!")


def compress() :

	response_files = sorted(glob.glob(IFILE['response'].format(d="*", t="*")))
	#print(response_files)

	# Allow for pending write operations
	time.sleep(1)

	# Compression 0:
	# zip-base64 contents

	print("COMPRESSION 0")

	# Brutal compression step
	for fn in response_files :
		continue
		#commons.zipjson_dump(commons.zipjson_load(fn), fn)


	# Compression I:
	# Remove records from file if present in the subsequent file

	print("COMPRESSION I: Remove duplicates in back-to-back records")

	for (fn1, fn2) in zip(response_files[:-1], response_files[1:]) :
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

		if not J1.intersection(J2) :
			continue

		J1 = J1.difference(J2)

		J1 = list(unhashable(list(J1)))
		J2 = list(unhashable(list(J2)))

		print("Compressing", fn1)
		commons.zipjson_dump(J1, fn1)


	# Compression II:
	# Remove route names if available elsewhere

	print("COMPRESSION II: Remove redundancies from individual records")

	# Route meta
	R = commons.zipjson_load(IFILE['routes'])

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
	assert(commons.is_distinct([g['RouteUID'] for g in R]))
	R = { g['RouteUID'] : g for g in R }

	def remove_single_route_redundancies(j) :

		subroute_id = j['SubRouteUID']
		assert(subroute_id in S)
		assert(j['Direction'] in S[subroute_id])
		s = S[subroute_id][j['Direction']]

		for key in ['SubRouteName', 'SubRouteID'] :
			if key in j :
				assert(j[key] == s[key])
				del j[key]

		if ('RouteUID' in j) :
			route_id = j['RouteUID']
			assert(route_id in R)
			r = R[route_id]

			for key in ['RouteName', 'RouteID'] :
				if key in j :
					assert(j[key] == r[key])
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

	def remove_global_route_redundancies(J) :
		if not J : return J

		assert(type(J) is list)

		def common_data(J) :

			# Set of common keys of entries of J
			# Typically, not all have the field 'BusStatus'
			K = sorted(set.intersection(*[set(j.keys()) for j in J]))

			for k in K :
				V = set(json.dumps(j[k]) for j in J)
				if (1 == len(V)) :
					yield (k, json.loads(V.pop()))

		C = dict(common_data(J))

		# print("Common key-values:", C)
		#
		# for (i, _) in enumerate(J) :
		# 	for k in C.keys() :
		# 		if k in J[i] :
		# 			pass
		# 			#del J[i][k]
		#
		# #J.append(C)

		return J

	for fn in response_files :
		try :
			J = commons.zipjson_load(fn)
		except EOFError :
			print("Warning: {} appears empty".format(fn))
			continue
		b = len(json.dumps(J)) # Before compression
		J = list(map(remove_single_route_redundancies, J))
		# J = remove_global_route_redundancies(J)
		a = len(json.dumps(J)) # After compression

		assert(a <= b)
		if (a == b) : continue

		print("Compressing", fn)
		commons.zipjson_dump(J, fn)


	print("DONE")


## ================== OPTIONS :

OPTIONS = {
	# 'DOWNLOAD' : download,
	'COMPRESS' : compress,
}


## ==================== ENTRY :

if (__name__ == "__main__") :

	assert(commons.parse_options(OPTIONS))