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

KEYS = {
	'busid': 'PlateNumb',

	'routeid': 'SubRouteUID',
	'dir': 'Direction',

	'time': 'GPSTime',
	'pos': 'BusPosition',
}

BUSID_OF = (lambda b: b[KEYS['busid']])

## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))


# Undo index_dicts_by_key on an element
def follow(b):
	# In case of a non-list, return a list with repeated element
	def listify(V):
		if type(V) is list:
			return V
		else:
			return [V] * len(b[KEYS['time']])

	# Note: the following does not work
	# assert(commons.is_distinct(b[key['time']]))
	# but duplicates will be eliminated later

	for x in zip(*[listify(b[k]) for k in KEYS.values()]):
		yield (dict(zip(KEYS.values(), x)))

	return

# Segment a list-like bb of bus records
# by route/direction
def segments(bb):
	# Put segment boundary when any of these keys change
	indicators = [KEYS['routeid'], KEYS['dir']]

	# Reverse list for easy pop-ing
	bb = list(reversed(list(bb)))

	while bb :
		# Initiate a new segment
		s = [bb.pop()]

		# Build segment while no indicators change
		while bb and all((bb[-1][k] == s[-1][k]) for k in indicators) :

			b = bb.pop()

			# None of the indicators have changed: continue the segment record

			# TODO: Segment if delta(GPSTime) is large?

			# However, if the timestamp is the same,
			if (b[KEYS['time']] == s[-1][KEYS['time']]):
				# ... then the rest of the record should be the same
				assert (b == s[-1])
				# Skip this redundant record
				continue

			s.append(b)

		# Sanity check: only one bus tracked in the segment
		assert(1 == len(set(map(BUSID_OF, s))))

		# Collapse into one record
		yield next(iter(commons.index_dicts_by_key(s, BUSID_OF).values()))

	return



## ===================== WORK :

def extract_busses() :

	response_files = sorted(glob.glob(IFILE['response'].format(d="20181102", t="1*")))
	time.sleep(1)

	# Load all bus records, group by Bus ID
	B = commons.index_dicts_by_key(
		chain.from_iterable(commons.zipjson_load(fn) for fn in response_files),
		BUSID_OF
	)

	i = 'KOffice_test'; B = { i : B[i] }

	print("Found {} physical busses".format(len(B)))

	#print(set(B[i][keys['routeid']])); exit(39)

	# Overview:
	# For each physical bus (identified by plate number)
	for b in B.values() :
		# Extract movement segments grouped by (SubRouteUID, Direction)
		for s in segments(follow(b)) :
			# Here, the segment s looks like
			# {'PlateNumb': '756-V2', 'SubRouteUID': 'KHH912', 'Direction': 1, 'GPSTime': [List of time stamps], 'BusPosition': [List of positions]}
			pass

	# The subkeys of keys['pos']
	keys_pos = {
		'Lat' : 'PositionLat',
		'Lon' : 'PositionLon',
	}

	# Function to compress the list of positions for a segment
	def unwrap_pos(s) :

		def listify(V) :
			return (V if (type(V) is list) else [V])

		s[KEYS['pos']] = listify(s[KEYS['pos']])

		(s[keys_pos['Lat']], s[keys_pos['Lon']]) = (
			list(coo) for coo in
			zip(*[tuple(bp[k] for k in keys_pos.values()) for bp in s[KEYS['pos']]])
		)
		del s[KEYS['pos']]

		# The position is a list; accord the time stamps, for convenience
		if PARAM['listify-timestamp'] :
			s[KEYS['time']] = listify(s[KEYS['time']])

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
