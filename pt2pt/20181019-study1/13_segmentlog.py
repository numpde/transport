#!/usr/bin/python3

# RA, 2018-10-31

## ================== IMPORTS :

from helpers import commons, maps

import uuid
import json
import inspect
import datetime as dt
import dateutil.parser
from itertools import chain, groupby, product
from joblib import Parallel, delayed


## ==================== NOTES :

# Use the same keys in KEYS and in IFILE/OFILE


## ================= FILE I/O :

open = commons.logged_open


## ================= METADATA :

# Keys of interest in the realtime bus network snapshot JSON record
KEYS = {
	'busid': 'PlateNumb',

	'routeid': 'SubRouteUID',
	'dir': 'Direction',

	'speed': 'Speed',
	'azimuth': 'Azimuth',

	'time': 'GPSTime',
	'pos': 'BusPosition',

	'busstatus' : 'BusStatus', # Not all records have this
	'dutystatus' : 'DutyStatus', # Or this
}

# The subkeys of KEYS['pos']
KEYS_POS = {
	'Lat': 'PositionLat',
	'Lon': 'PositionLon',
}

# Helpers
BUSID_OF = (lambda b: b[KEYS['busid']])
ROUTEID_OF = (lambda r: r[KEYS['routeid']])
DIRECTION_OF = (lambda r: r[KEYS['dir']])

# What finally identifies a one-way route
RUN_KEY = (lambda r: (ROUTEID_OF(r), DIRECTION_OF(r)))


## ================== INPUT 1 :

IFILE = {
	'realtime_log_file' : "OUTPUT/12/Kaohsiung/UV/{d}/{t}.json",
}


## ==================== PARAM :

PARAM = {
	'datetime_filter' : {
		# "Scenario" -- output subdirectory
		'path': "Kaohsiung/20181105-20181111",
		# datetime filter
		'func': (lambda t : (t.year == 2018) and (t.month == 11) and (5 <= t.day <= 11)),
	},

	'mapbox_api_token' : open(".credentials/UV/mapbox-token.txt", 'r').read(),

	# Sort snapshots by GPS time
	'segment_sort_by_gpstime' : True,

	# Execute run segmentation in parallel
	'segment_parallel_jobs' : 4,

	# Segment runs whenever the timegap between measurements is large
	'segment_timegap_minutes': 5,

	# Segment runs whenever any of these fields change
	'segment_indicators' : set(RUN_KEY({v : v for v in KEYS.values()})) | {KEYS['busstatus'], KEYS['dutystatus']},

	# Drop records without movement
	'drop_stationary' : True,

	# Do not replace monotonous lists with singletons, or with the value
	'keys_dont_collapse' : [KEYS[k] for k in ['pos', 'speed', 'azimuth', 'time']],

	# Leave those fields as singletons
	'keys_singletons_ok' : [],

	# Quality filters for run-by-route
	'quality_min_run_waypoints' : 4,
	'quality_min_diameter' : 300, # (meters)
	'quality_only_normal_status' : True,

	# Attach a quality indicator for each run
	'quality_key' : 'quality', # Note: run['quality'] in ['+', '-']
}


## =================== OUTPUT :

OFILE = {
	'segment_by_bus' :   "OUTPUT/13/" + PARAM['datetime_filter']['path'] + "/bybus/UV/{busid}.json",

	'segment_by_route' : "OUTPUT/13/" + PARAM['datetime_filter']['path'] + "/byroute/UV/{routeid}-{dir}.{{ext}}",
}

commons.makedirs(OFILE)


## ================== INPUT 2 :

IFILE.update({
	'segment_by_bus': OFILE['segment_by_bus'],
})


## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))


# GPS timestamp reader
gpstime = (lambda b: dateutil.parser.parse(b[KEYS['time']]))


# Retain only the known fields and simplify structure
def drop_fields(b) :
	# Simplify the geo-position field
	b[KEYS['pos']] = (b[KEYS['pos']][KEYS_POS['Lat']], b[KEYS['pos']][KEYS_POS['Lon']])
	# Keep only the registered fields
	return { k : b[k] for k in KEYS.values() if k in b }


# Segment a list-like bb of bus records by route/direction
def segments(bb):
	# Sort records by GPS time
	if PARAM['segment_sort_by_gpstime'] :
		bb = sorted(bb, key=gpstime)

	# Reverse list for easy pop-ing
	bb = list(reversed(list(bb)))

	while bb :
		# Initiate a new segment
		s = [bb.pop()]

		# Build segment while no indicators change
		while bb and all((bb[-1].get(k, '?') == s[-1].get(k, '?')) for k in PARAM['segment_indicators']) :

			# None of the indicators have changed: continue the segment record

			(t0, t1) = (gpstime(b) for b in [s[-1], bb[-1]])
			assert(t0 <= t1), "The records should be sorted by GPS time"

			# ... unless there is a large time gap
			if ((t1 - t0) > dt.timedelta(minutes=PARAM['segment_timegap_minutes'])) :
				break

			b = bb.pop()

			# If the timestamp is the same ignore this record
			# Note: sometimes the rest of the fields may change,
			# in particular the location (even over 100m),
			# possibly due to the GPS-time field precision of 1 sec
			if (t0 == t1): continue

			s.append(b)

		# Sanity check: only one bus tracked in the segment
		assert(1 == len(set(map(BUSID_OF, s))))

		# Collapse into one record
		run = next(iter(
			commons.index_dicts_by_key(
				s, BUSID_OF, keys_dont_collapse=PARAM['keys_dont_collapse'], keys_singletons_ok=PARAM['keys_singletons_ok']
			).values()
		))

		# Attach an ad-hoc ID tag
		run['RunUUID'] = uuid.uuid4().hex

		if PARAM['drop_stationary'] :
			if (1 == len(set(run[KEYS['pos']]))) :
				continue

		yield run


# Check if 'values' only contains "normal" values
def is_normal_status(values, normal_values=[0, '0']) :
	if type(values) in [list, set] :
		return all(is_normal_status(v) for v in values)
	else :
		return (values in normal_values)

def diameter(pp) :
	return max(commons.geodesic(p, q) for (p, q) in product(pp, repeat=2))

def is_run_acceptable(run) :
	# Trivial or too-short run
	if (len(run[KEYS['pos']]) < PARAM['quality_min_run_waypoints']) : return False

	# Consistently of "Normal" status?
	if not is_normal_status(run.get(KEYS['busstatus'], 0)) : return False
	if not is_normal_status(run.get(KEYS['dutystatus'], 0)) : return False

	# Is there sufficient movement?
	if (diameter(run[KEYS['pos']]) < PARAM['quality_min_diameter']) : return False

	return True


## =================== SLAVES :

def segment_by_bus() :

	# Return bus logs one by one from realtime log files
	def read_realtime_logs(filenames) :
		for fn in filenames :
			try :
				for r in commons.zipjson_load(fn) :
					yield r
			except json.decoder.JSONDecodeError as e :
				print("Warning: could not read {} ({})".format(fn, e))

	# Collect realtime log filenames
	logs = [
		fn
		for fn in commons.ls(IFILE['realtime_log_file'].format(d="*", t="*"))
		if PARAM['datetime_filter']['func'](dt.datetime.strptime(fn, IFILE['realtime_log_file'].format(d="%Y%m%d", t="%H%M%S")))
	]

	print("Collecting bus ids...")

	busids = set(map(BUSID_OF, read_realtime_logs(logs)))
	print("{} buses: {}".format(len(busids), busids))

	def process_busid(busid) :

		J = list(segments(drop_fields(r) for r in read_realtime_logs(logs) if (BUSID_OF(r) == busid)))

		# Note: A segment looks like this (some fields omitted)
		# {'PlateNumb': '159-XH', 'SubRouteUID': 'KHH1421', 'Direction': 1, 'Speed': [0.0, 0.0, 0.0, 24.0, 35.0, 0.0, 6.0, 54.0, 0.0, 9.0, 42.0, 0.0, 0.0, 0.0, 0.0, 6.0, 13.0, 0.0, 0.0, 0.0, 18.0, 22.0, 6.0, 0.0, 23.0, 35.0, 0.0, 17.0, 6.0, 0.0, 9.0, 13.0, 0.0, 18.0, 25.0, 14.0, 0.0, 9.0, 5.0, 0.0, 0.0, 13.0, 5.0, 21.0, 8.0, 0.0, 21.0, 29.0, 38.0, 0.0, 0.0, 8.0, 39.0, 0.0, 0.0, 25.0, 0.0, 17.0, 31.0, 38.0, 6.0, 0.0, 5.0, 0.0, 0.0], 'Azimuth': [81.0, 81.0, 81.0, 70.0, 72.0, 70.0, 53.0, 70.0, 69.0, 70.0, 70.0, 78.0, 78.0, 78.0, 76.0, 11.0, 65.0, 71.0, 67.0, 72.0, 68.0, 75.0, 91.0, 94.0, 71.0, 70.0, 66.0, 28.0, 355.0, 6.0, 7.0, 267.0, 258.0, 261.0, 23.0, 21.0, 22.0, 353.0, 329.0, 323.0, 323.0, 352.0, 334.0, 329.0, 342.0, 355.0, 348.0, 321.0, 306.0, 299.0, 299.0, 36.0, 31.0, 30.0, 30.0, 4.0, 357.0, 352.0, 351.0, 1.0, 279.0, 279.0, 277.0, 331.0, 331.0], 'GPSTime': ['2018-11-05T09:59:35+08:00', '2018-11-05T09:59:55+08:00', '2018-11-05T10:00:35+08:00', '2018-11-05T10:01:15+08:00', '2018-11-05T10:01:39+08:00', '2018-11-05T10:01:55+08:00', '2018-11-05T10:02:57+08:00', '2018-11-05T10:03:55+08:00', '2018-11-05T10:04:35+08:00', '2018-11-05T10:05:35+08:00', '2018-11-05T10:05:55+08:00', '2018-11-05T10:06:15+08:00', '2018-11-05T10:06:55+08:00', '2018-11-05T10:07:35+08:00', '2018-11-05T10:07:55+08:00', '2018-11-05T10:08:23+08:00', '2018-11-05T10:09:15+08:00', '2018-11-05T10:09:35+08:00', '2018-11-05T10:09:55+08:00', '2018-11-05T10:10:55+08:00', '2018-11-05T10:11:03+08:00', '2018-11-05T10:11:35+08:00', '2018-11-05T10:12:15+08:00', '2018-11-05T10:12:35+08:00', '2018-11-05T10:13:25+08:00', '2018-11-05T10:13:39+08:00', '2018-11-05T10:14:15+08:00', '2018-11-05T10:18:35+08:00', '2018-11-05T10:21:03+08:00', '2018-11-05T10:21:35+08:00', '2018-11-05T10:22:15+08:00', '2018-11-05T10:22:40+08:00', '2018-11-05T10:22:55+08:00', '2018-11-05T10:23:35+08:00', '2018-11-05T10:24:00+08:00', '2018-11-05T10:24:55+08:00', '2018-11-05T10:25:15+08:00', '2018-11-05T10:25:55+08:00', '2018-11-05T10:26:15+08:00', '2018-11-05T10:26:35+08:00', '2018-11-05T10:27:35+08:00', '2018-11-05T10:27:55+08:00', '2018-11-05T10:28:35+08:00', '2018-11-05T10:29:00+08:00', '2018-11-05T10:29:19+08:00', '2018-11-05T10:29:55+08:00', '2018-11-05T10:30:15+08:00', '2018-11-05T10:30:43+08:00', '2018-11-05T10:33:01+08:00', '2018-11-05T10:33:35+08:00', '2018-11-05T10:33:55+08:00', '2018-11-05T10:34:39+08:00', '2018-11-05T10:34:59+08:00', '2018-11-05T10:35:55+08:00', '2018-11-05T10:36:15+08:00', '2018-11-05T10:36:57+08:00', '2018-11-05T10:37:15+08:00', '2018-11-05T10:38:05+08:00', '2018-11-05T10:38:16+08:00', '2018-11-05T10:38:39+08:00', '2018-11-05T10:39:15+08:00', '2018-11-05T10:39:35+08:00', '2018-11-05T10:40:35+08:00', '2018-11-05T10:40:55+08:00', '2018-11-05T10:41:35+08:00'], 'BusPosition': [(22.615449, 120.298779), (22.615449, 120.298779), (22.615449, 120.298779), (22.615909, 120.300139), (22.616379, 120.30158), (22.61655, 120.302109), (22.6168, 120.302549), (22.61792, 120.306129), (22.618179, 120.30695), (22.618539, 120.30812), (22.61899, 120.30951), (22.61923, 120.310299), (22.61923, 120.310299), (22.619579, 120.31152), (22.619659, 120.311819), (22.619879, 120.31198), (22.620149, 120.312939), (22.62046, 120.31389), (22.620499, 120.314049), (22.62058, 120.31443), (22.62065, 120.314729), (22.62142, 120.316739), (22.62169, 120.317409), (22.621719, 120.317599), (22.622309, 120.318939), (22.62264, 120.32008), (22.62311, 120.32161), (22.629729, 120.32777), (22.63231, 120.328199), (22.632509, 120.32818), (22.632699, 120.328239), (22.63294, 120.32792), (22.632899, 120.32742), (22.632799, 120.32657), (22.633639, 120.326419), (22.63803, 120.32765), (22.63812, 120.32769), (22.63997, 120.32786), (22.64025, 120.327789), (22.640619, 120.327569), (22.640619, 120.327569), (22.64175, 120.327129), (22.643129, 120.32653), (22.644439, 120.32598), (22.645409, 120.32563), (22.64553, 120.32563), (22.64656, 120.325509), (22.64882, 120.325059), (22.651849, 120.321129), (22.65296, 120.31946), (22.65296, 120.31946), (22.654109, 120.319339), (22.65536, 120.320199), (22.6569, 120.321259), (22.6569, 120.321259), (22.65991, 120.322019), (22.66066, 120.322069), (22.661239, 120.322029), (22.661999, 120.321929), (22.664059, 120.32182), (22.66559, 120.3216), (22.66559, 120.321579), (22.66562, 120.32138), (22.665659, 120.3212), (22.665659, 120.3212)], 'RunUUID': '2f9611d6ceb241558c2903e644ace77c'}

		with open(OFILE['segment_by_bus'].format(busid=busid), 'w') as fd :
			json.dump(J, fd)

	Parallel(n_jobs=PARAM['segment_parallel_jobs'])(delayed(process_busid)(busid) for busid in sorted(busids))


def segment_by_route() :

	# A "case" is the result of "RUN_KEY", i.e. a pair (routeid, direction)

	print("Collecting cases...")

	# Associate to each case a list of files that contain instances of it
	case_directory = {
		case : set( r[1] for r in g )
		for (case, g) in groupby(
			sorted(
				(RUN_KEY(s), busfile)
				for busfile in commons.ls(IFILE['segment_by_bus'].format(busid="*"))
				for s in commons.zipjson_load(busfile)
			),
			key=(lambda r : r[0])
		)
	}

	for (case, files) in sorted(case_directory.items(), key=(lambda cf : -len(cf[1]))) :
		segments = [
			{
				**s,
				PARAM['quality_key'] : ("+" if is_run_acceptable(s) else "-")
			}
			for busfile in files
			for s in commons.zipjson_load(busfile)
			if (RUN_KEY(s) == case)
		]

		if not segments :
			print("Warning: No valid bus runs found for {}".format(case))
			continue

		fn = OFILE['segment_by_route'].format(**{k : segments[0].get(K) for (k, K) in KEYS.items()})

		with open(fn.format(ext="json"), 'w') as fd :
			json.dump(segments, fd)

		with open(fn.format(ext="png"), 'wb') as fd :
			waypoints_by_quality = { q : list(s[KEYS['pos']] for s in g) for (q, g) in groupby(segments, key=(lambda s : s[PARAM['quality_key']])) }
			maps.write_track_img(waypoints=list(chain.from_iterable(waypoints_by_quality.get('-', []))), tracks=waypoints_by_quality.get('+', []), fd=fd, mapbox_api_token=PARAM['mapbox_api_token'])


## =================== MASTER :

def segment_logs() :
	segment_by_bus()
	segment_by_route()


## ================== OPTIONS :

OPTIONS = {
	'SEGMENT' : segment_logs,
	'SEGMENT_BY_BUS' : segment_by_bus,
	'SEGMENT_BY_ROUTE' : segment_by_route,
}


## ==================== ENTRY :

if (__name__ == "__main__") :
	commons.parse_options(OPTIONS)
