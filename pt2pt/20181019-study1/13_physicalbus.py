#!/usr/bin/python3

# RA, 2018-10-31

## ================== IMPORTS :

import commons

import time
import json
import glob
import inspect
from itertools import chain


## ==================== NOTES :

pass


## ==================== INPUT :

IFILE = {
	'response' : "OUTPUT/12/Kaohsiung/UV/{d}/{t}.json",

	#'route-stops' : "ORIGINALS/MOTC/Kaohsiung/CityBusApi_StopOfRoute/data.json",
}


## =================== OUTPUT :

OFILE = {
	'busses' : "OUTPUT/13/Kaohsiung/UV/{busid}.json",
}

commons.makedirs(OFILE)

## ==================== PARAM :

PARAM = {
	'listify-timestamp' : True,
}

## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))


## ===================== WORK :

def extract_busses() :

	response_files = sorted(glob.glob(IFILE['response'].format(d="20181102", t="1*")))
	time.sleep(1)

	keys = {
		'busid'   : 'PlateNumb',

		'routeid' : 'SubRouteUID',
		'dir'     : 'Direction',

		'time'    : 'GPSTime',
		'pos'     : 'BusPosition',
	}

	busid_of = (lambda b : b[keys['busid']])

	# Load all bus records, group by Bus ID
	B = commons.index_dicts_by_key(
		chain.from_iterable(commons.zipjson_load(fn) for fn in response_files),
		busid_of
	)

	i = 'KOffice_test'
	B = { i : B[i] }

	print("Found {} physical busses".format(len(B)))

	for b in B:
		print(b)

	# Undo index_dicts_by_key on an element
	def follow(b) :

		# In case of a non-list, make an iterable returning this element
		def listify(V):
			if type(V) is list:
				return V
			else:
				while True: yield V

		# TODO: the following does not hold
		#assert(commons.is_distinct(b[key_time]))

		for x in zip(*[ listify(b[k]) for k in keys.values() ]) :
			yield(dict(zip(keys.values(), x)))

	def segments(bb) :

		# Put segment boundary when any of these keys change
		indicators = [keys['routeid'], keys['dir']]

		# Current segment
		L = []

		for b in bb :
			# if (busid_of(b) == 'KOffice_test') :
			# 	print(b)
			if not L :
				# First record of the new segment
				L.append(b)
			elif all((b[k] == L[-1][k]) for k in indicators) :
				# None of the indicators have changed, so
				# continue the segment record
				# TODO: Segment if delta(GPSTime) is large?

				# (unless there is no change at all, then do nothing)
				if (b == L[-1]) :
					continue

				L.append(b)
			else :
				# Sanity check: only one bus tracked in the segment
				assert(1 == len(set(busid_of(b) for b in L)))

				# Collapse into one record
				b = next(iter(commons.index_dicts_by_key(L, busid_of).values()))

				# Initiate new segment
				L = []

				yield b

	# Overview:
	# For each physical bus (identified by plate number)
	for b in B.values() :
		# Extract movement segments grouped by (SubRouteUID, Direction)
		for s in segments(follow(b)) :
			# Here, the segment s looks like
			# {'PlateNumb': '756-V2', 'SubRouteUID': 'KHH912', 'Direction': 1, 'GPSTime': [List of time stamps], 'BusPosition': [List of positions]}
			print(s)
			pass

	exit(39)

	keys_pos = {
		'Lat' : 'PositionLat',
		'Lon' : 'PositionLon',
	}

	# Function to compress the list of positions for a segment
	def unwrap_pos(s) :
		if not (type(s[keys['pos']]) is list) :
			s[keys['pos']] = list(s[keys['pos']])

		(s[keys_pos['Lat']], s[keys_pos['Lon']]) = (
			list(coo) for coo in
			zip(*[ tuple(bp[k] for k in keys_pos.values()) for bp in s[keys['pos']] ])
		)
		del s[keys['pos']]

		# The position is a list; accord the time stamps, for convenience
		if PARAM['listify-timestamp'] :
			s[keys['time']] = listify(s[keys['time']])

		return s

	# Dump info to disk
	for (busid, b) in B.items() :
		J = list(unwrap_pos(s) for s in segments(follow(b)))
		fn = OFILE['busses'].format(busid=busid)
		with open(fn, 'w') as fd :
			json.dump(J, fd)


## ===================== PLAY :

pass


## ================== OPTIONS :

OPTIONS = {
	'EXTRACT_BUSSES' : extract_busses,
}

## ==================== ENTRY :

if (__name__ == "__main__") :

	assert(commons.parse_options(OPTIONS))
