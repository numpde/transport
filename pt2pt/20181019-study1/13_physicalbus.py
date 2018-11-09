#!/usr/bin/python3

# RA, 2018-10-31

## ================== IMPORTS :

from helpers import commons

import time
import json
import glob
import inspect
import datetime as dt
from itertools import chain


## ==================== NOTES :

pass


## ==================== INPUT :

IFILE = {
	'response' : "OUTPUT/12/Kaohsiung/UV/{d}/{t}.json",
}


## =================== OUTPUT :

OFILE = {
	'busses' : "OUTPUT/13/Kaohsiung/UV/{busid}.json",
}

commons.makedirs(OFILE)

## ================= METADATA :

# Keys in a response file JSON record
KEYS = {
	'busid': 'PlateNumb',

	'routeid': 'SubRouteUID',
	'dir': 'Direction',

	'speed': 'Speed',
	'azimuth': 'Azimuth',

	'time': 'GPSTime',
	'pos': 'BusPosition',

	#'bus_stat' : 'BusStatus', # Not all records have this
	#'duty_stat' : 'DutyStatus',
}

# The subkeys of KEYS['pos']
KEYS_POS = {
	'Lat': 'PositionLat',
	'Lon': 'PositionLon',
}

# Helper to extract the Physical-Bus ID
BUSID_OF = (lambda b: b[KEYS['busid']])


## ==================== PARAM :

PARAM = {
	'listify-keys' : ['speed', 'azimuth', 'time'],
}


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
	# assert(commons.all_distinct(b[key['time']]))
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
				if not (b == s[-1]) :
					# Timestamp is the same but the complete record is not
					pos = (lambda r : (r[KEYS['pos']][KEYS_POS['Lat']], r[KEYS['pos']][KEYS_POS['Lon']]))
					d = commons.geodesic(pos(b), pos(s[-1]))
					# Grace of a few meters: displacement + GPS inaccuracy
					assert(d <= 200)
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

	response_files = sorted(glob.glob(IFILE['response'].format(d="20181103", t="0*")))
	time.sleep(1)

	# Load all bus records, group by Bus ID
	B = commons.index_dicts_by_key(
		chain.from_iterable(commons.zipjson_load(fn) for fn in response_files),
		BUSID_OF
	)

	print("Found {} physical busses".format(len(B)))

	# Sanity check: forward-only GPSTime
	for b in B.values() :
		T = b[KEYS['time']]
		if not (T is list) : continue
		T = [dt.datetime.fromisoformat(t) for t in T]
		assert(all((s <= t) for (s, t) in zip(T[:-1], T[1:])))

	# Overview:
	# For each physical bus (identified by plate number)
	for b in B.values() :
		# Extract movement segments grouped by (SubRouteUID, Direction)
		for s in segments(follow(b)) :
			# Here, the segment s looks like
			# {'PlateNumb': '756-V2', 'SubRouteUID': 'KHH912', 'Direction': 1, 'GPSTime': [List of time stamps], 'BusPosition': [List of positions]}
			pass

	# Function to "unwrap" the list of positions for a segment
	def unwrap_pos(s) :

		listify = (lambda V : V if (type(V) is list) else [V])

		(s[KEYS_POS['Lat']], s[KEYS_POS['Lon']]) = (
			list(coo) for coo in
			zip(*[
				tuple(bp[k] for k in KEYS_POS.values()) for bp in listify(s[KEYS['pos']])
			])
		)

		del s[KEYS['pos']]

		for k in PARAM['listify-keys'] :
			s[KEYS[k]] = listify(s[KEYS[k]])

		return s

	# Dump info to disk
	for (busid, b) in B.items() :
		J = list(unwrap_pos(s) for s in segments(follow(b)))
		with open(OFILE['busses'].format(busid=busid), 'w') as fd :
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
