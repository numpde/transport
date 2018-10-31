#!/usr/bin/python3

# RA, 2018-10-31

## ================== IMPORTS :

import os
import sys
import time
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

	# Allow for pending write operations
	time.sleep(1)

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

	def remove_single_route_redundancies(j) :

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
		J = json.load(open(fn, 'r'))
		b = len(json.dumps(J))
		J = list(map(remove_single_route_redundancies, J))
		# J = remove_global_route_redundancies(J)
		a = len(json.dumps(J))

		assert(a <= b)
		if (a == b) : continue

		print("Compressing", fn)
		json.dump(J, open(fn, 'w'))


def rerunbus() :

	response_files = sorted(glob.glob(IFILE['response'].format(dt="*")))
	time.sleep(1)

	import matplotlib.pyplot as plt

	plt.ion()
	plt.show()

	BP = { }

	for fn in response_files :

		J = json.load(open(fn, 'r'))

		# Filter down to one route
		J = [j for j in J if (j['SubRouteUID'] in ['KHH122', 'KHH1221', 'KHH882'])]

		if not J : continue

		try :
			# The plate number should be unique
			assert(len(J) == len(set(j['PlateNumb'] for j in J)))
		except AssertionError :
			# It is not always the case!
			for j in sorted(J, key=(lambda j : j['PlateNumb'])) :
				print(j)
			raise

		# Index by the plate number
		J = { j['PlateNumb'] : j for j in J }

		BP_before = BP

		BP = {
			pn : (bp['PositionLon'], bp['PositionLat'])
			for (pn, bp) in [
				(pn, j['BusPosition']) for (pn, j) in J.items()
			]
		}

		s = { pn : {0: 'r', 90: 'g', 98: 'm', 99: 'k'}[int(J[pn]['BusStatus'])] for pn in BP.keys() }

		for pn in set.intersection(set(BP.keys()), set(BP_before.keys())) :
			plt.plot(*zip(BP_before[pn], BP[pn]), '-' + s[pn], linewidth=0.1)
			pass

		(x, y) = zip(*BP.values())
		h = plt.scatter(x, y, c=list(s.values()))

		plt.draw()
		plt.pause(0.1)

		h.remove()

		#plt.plot(*zip(*BP.values()), 'b.')

	time.sleep(5)
	input("Please press ENTER")


## ================== OPTIONS :

OPTIONS = {
	'DOWNLOAD' : download,
	'COMPRESS' : compress,
	'RERUNBUS' : rerunbus,
}

def parse_options() :

	if (len(sys.argv) > 1) :
		(OPT, ARG) = (sys.argv[1], sys.argv[2:])
		for (opt, fun) in OPTIONS.items() :
			if (opt == OPT) :
				fun(*ARG)
				return True

		raise RuntimeError("Unrecognized command line option")

	return False


## ==================== ENTRY :

if (__name__ == "__main__") :

	if not parse_options() :

		print("Please specify option via command line:", *OPTIONS.keys())

