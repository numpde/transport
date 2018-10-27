#!/usr/bin/python3

# RA, 2018-10-26

## ================== IMPORTS :

import os
import sklearn.neighbors
import geopy.distance
import numpy as np
import builtins
import json
import inspect
import pickle
import time
import urllib.request
from collections import defaultdict

## ==================== NOTES :

pass


## ==================== INPUT :

IFILE = {
	'request-routes' : "request_cache/routes_{lang}.json",
	'request-stops'  : "request_cache/stops_{ID}-{Dir}_{lang}.json",

	'compute-knn'    : "compute_cache/UV/stops-knn.pkl",
}


## =================== OUTPUT :

OFILE = {
	'request-routes' : IFILE['request-routes'],
	'request-stops'  : IFILE['request-stops'],

	'compute-knn'    : IFILE['compute-knn'],
}

# Create output directories
for f in OFILE.values() : os.makedirs(os.path.dirname(f), exist_ok=True)


## ==================== PARAM :

PARAM = {
	'logged-open' : True,

	'wget-max-calls': 5,
	'wget-throttle-seconds' : 1,
	'wget-always-reuse-file' : True,

	'url-routes' : {
		'en' : "https://ibus.tbkc.gov.tw/KSBUSN/NewAPI/RealRoute.ashx?type=GetRoute&Lang=En",
		'tw' : "https://ibus.tbkc.gov.tw/KSBUSN/NewAPI/RealRoute.ashx?type=GetRoute&Lang=Cht",
	},

	'url-routestops' : {
		'en' : "https://ibus.tbkc.gov.tw/KSBUSN/NewAPI/RealRoute.ashx?type=GetStop&Data={ID}_,{Dir}&Lang=En",
		'tw' : "https://ibus.tbkc.gov.tw/KSBUSN/NewAPI/RealRoute.ashx?type=GetStop&Data={ID}_,{Dir}&Lang=Cht",
		# Note: 'Dir' is the direction there=1 and back=2, elsewhere keyed by 'GoBack' or 'Goback' instead
	},

	'try-local-routes' : True,
	'try-local-routestops' : True,

	'force-recompute-knn' : False,
}


## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))

# Log which files are opened
def logged_open(filename, mode='r', *argv, **kwargs) :
	print("({}):\t{}".format(mode, filename))
	return builtins.open(filename, mode, *argv, **kwargs)

# Activate this function?
if PARAM.get('logged-open') : open = logged_open

# Class to fetch files via HTTP
class wget :

	number_of_calls = 0

	def __init__(self, url, filename=None) :

		if filename and PARAM['wget-always-reuse-file'] :
			if os.path.isfile(filename) :
				return

		wget.number_of_calls = wget.number_of_calls + 1

		if (wget.number_of_calls > PARAM['wget-max-calls']) :
			raise RuntimeError("Call limit exceeded for wget")

		time.sleep(PARAM['wget-throttle-seconds'])

		with urllib.request.urlopen(url) as response :
			if filename :
				with open(filename, 'wb') as f :
					f.write(response.read())
			else :
				self.bytes = response.read()

# Index a list _I_ of dict's by the return value of key_func
def reindex_by_key(I, key_func) :
	J = defaultdict(lambda: defaultdict(list))

	for i in I :
		for (k, v) in i.items() :
			J[key_func(i)][k].append(v)

	# Convert all defaultdict to dict
	J = json.loads(json.dumps(J))

	return J

# "Turn" a JSON structure, rekey by id_key
# Assume: I[lang] is a list of dict's, where each has a field id_key
def lang_reform(I, id_key) :

	J = defaultdict(lambda: defaultdict(dict))

	for (lang, E) in I.items() :
		for e in E :
			for (k, v) in e.items() :
				J[e[id_key]][k][lang] = v

	for (i, e) in J.items() :
		for (k, V) in e.items() :
			V = V.values()
			if (len(set(V)) == 1) :
				J[i][k] = set(V).pop()

	# Convert all defaultdict to dict
	J = json.loads(json.dumps(J))

	return J

# Metric for (lat, lon) coordinates
def geodesic(a, b) :
	return geopy.distance.geodesic(a, b).m


## ================== CLASSES :

class RoutesMeta :

	def routes_init(self) :

		routes_by_lang = { }

		for (lang, url) in PARAM['url-routes'].items() :

			filename = IFILE['request-routes'].format(lang=lang)

			if PARAM['force-fetch-request-routes'] :
				with open(filename, 'wb') as f :
					f.write(wget(url).bytes)
			else:
				wget(url, filename)

			with open(filename, 'r') as f :
				routes_by_lang[lang] = json.load(f)

		self.routes = lang_reform(routes_by_lang, 'ID')

		# An entry of self.routes now looks like this:
		# (assuming route_id_key is 'ID')
		#
		# self.routes['1431'] == {
		#     'ID': '1431',
		#     'nameZh': {'en': '0 North', 'tw': '0北'},
		#     'gxcode': '001',
		#     'ddesc': {'en': 'Golden Lion Lake Station<->Golden Lion Lake Station', 'tw': '金獅湖站－金獅湖站'},
		#     'departureZh': {'en': 'Golden Lion Lake Station', 'tw': '金獅湖站'},
		#     'destinationZh': {'en': 'MRT Yanchengpu Station', 'tw': '捷運鹽埕埔站'},
		#     'RouteType': {'en': 'General Line', 'tw': '一般公車'},
		#     'MasterRouteName': ' ',
		#     'MasterRouteNo': '0',
		#     'MasterRouteDesc': {'en': 'MRT Yanchengpu Station<->MRT Yanchengpu Station', 'tw': '捷運鹽埕站－捷運鹽埕站'},
		#     'routes': '2',
		#     'ProviderName': {'en': 'Han Cheng Bus Company', 'tw': '漢程客運'},
		#     'ProviderWebsite': 'http://www.ibus.com.tw/city-bus/khh/',
		#     'TimeTableUrl': 'http://ibus.com.tw/timetable/0N.pdf'
		# }

	def __init__(self) :
		self.routes_init()
		assert(self.routes)


class Stops :

	def init(self, routesmeta) :

		# self.trajs[route_id][direction] is a list of stop SIDs
		self.routes = defaultdict(dict)
		# self.stops is a dict of all stops/platforms keyed by SID
		self.stops = defaultdict(dict)

		for (lang, preurl) in PARAM['url-routestops'].items():
			for (i, route) in routesmeta.routes.items() :

				nroutes = int(route['routes'])
				assert(nroutes in [1, 2])

				self.routes[i] = route
				self.routes[i]['Dir'] = { }

				for Dir in [1, 2][:nroutes] :

					url = preurl.format(ID=i, Dir=Dir)
					filename = IFILE['request-stops'].format(ID=i, Dir=Dir, lang=lang)

					if (not os.path.isfile(filename)) or (not PARAM['try-local-routestops']) :
						wget(url, filename)

					# Special cases:
					#
					# As of 2018-10-27, the following routes get an error response:
					# 1602-1, 2173-2, 2482-1, 371-1
					#
					# http://southeastbus.com/index/kcg/Time/248.htm
					if (i == '2482') and (Dir == 1) : continue
					# https://www.crowntaxi.com.tw/news.aspx?ID=49
					if (i == '331') and (Dir == 1) : continue
					# http://southeastbus.com/index/kcg/Time/37.htm
					if (i == '371') and (Dir == 1) : continue
					# 'http://southeastbus.com/index/kcg/Time/O7A.htm
					if (i == '1602') and (Dir == 1) : continue

					with open(filename, 'r') as f :
						try :
							J = json.load(f)
						except json.decoder.JSONDecodeError as e :
							print(route)
							raise

					if not (type(J) is list) :
						raise RuntimeWarning("Expect a list in JSON response here")

					# The SIDs of stops along this route will be written here in correct order
					self.routes[i]['Dir'][Dir] = []

					for stop in sorted(J, key=(lambda r : int(r['seqNo']))) :

						n = stop['SID']

						self.routes[i]['Dir'][Dir].append(n)

						del stop['seqNo']

						# Special cases:
						if True :

							# Presumably error in data in route 2482
							# Assign Id of 'MRT Wukuaicuo Station (Jhongjheng 1st Rd.)'
							if (n == '4536') : stop['Id'] = '0062'
							# Assign id of 'MRT Martial Arts Stadium Station'
							if (n == '3522') : stop['Id'] = '0004'

							# Special case (apparent data error in route 351-2)
							if (n == '15883') : stop['Id'] = '3215'

							# In route 50-2
							# Resolve 'Id' of 'MRT Sizihwan Station (LRT Hamasen)'
							if (n == '2120') : stop['Id'] = '9203'

							# In route 701-2
							# Resolve 'Id' of 'Chengcing Lake Baseball Field'
							if (n == '2542') : stop['Id'] = '3535'

							# In route 7-1 and 7-2
							# Resolve 'Id' of 'Houjin Junior High School (MRT Houjin Station)'
							if (n == '4583') : stop['Id'] = '7010'
							if (n == '4680') : stop['Id'] = '7010'

							# In route 731-1
							# Resolve 'Id' of 'MRT Metropolitan Park Station'
							if (n == '5135') : stop['Id'] = '7508'

							# ETC... WHATEVER, WE SIMPLY NULLIFY THE 'Id' FIELD

							stop['Id'] = None

						if n in self.stops[lang] :
							if not (self.stops[lang][n] == stop) :
								print("Ex. A:", self.stops[lang][n])
								print("Ex. B:", stop)
								raise RuntimeError("Data inconsistency")
						else :
							self.stops[lang][n] = stop

		# Rekey by the Stop ID
		self.stops = { lang : stops.values() for (lang, stops) in self.stops.items() }
		self.stops = lang_reform(self.stops, 'SID')


		# Now, to each stop append the list of incident routes

		for n in self.stops.keys() :
			self.stops[n]['routes'] = defaultdict(list)

		for (i, r) in self.routes.items() :
			for (Dir, stops) in r['Dir'].items() :
				for n in stops :
					self.stops[n]['routes'][Dir].append(i)
		
		# Convert all defaultdict to dict
		self.stops = json.loads(json.dumps(self.stops))

	def __init__(self, R) :

		assert(type(R) is RoutesMeta), "Type checking failed"
		self.init(R)


class StopsKNN(Stops) :

	def __init__(self, R) :
		Stops.__init__(self, R)
		self.init_knn()

	def init_knn(self) :

		if PARAM['force-recompute-knn'] or (not os.path.isfile(IFILE['compute-knn'])) :

			(I, X) = zip(*[ (i, (float(s['latitude']), float(s['longitude']))) for (i, s) in self.stops.items() ])

			self.knn = {
				'SIDs' : I,
				'tree' : sklearn.neighbors.BallTree(X, leaf_size=30, metric='pyfunc', func=geodesic),
			}

			with open(OFILE['compute-knn'], 'wb') as f :
				pickle.dump(self.knn, f, pickle.HIGHEST_PROTOCOL)

		else :

			try :

				with open(IFILE['compute-knn'], 'rb') as f :
					self.knn = pickle.load(f)

			except EOFError as e :

				PARAM['force-recompute-knn'] = True
				self.init_knn()

	def get_nearest_stops(self, pos, k) :

		# Note: assume a single sample pos, i.e. pos = (lat, lon)

		(dist, ind) = self.knn['tree'].query(np.asarray(pos).reshape(1, -1), k=k)
		(dist, ind) = (dist.flatten(), ind.flatten())

		# Convert ind to stop IDs
		ind = [ self.knn['SIDs'][n] for n in ind ]

		# Get the complete nearest stops info
		stops = [ self.stops[j] for j in ind ]

		# Append the 'distance' info to each nearest stop
		for k in range(len(stops)) :
			stops[k]['distance'] = dist[k]

		# Index stops by ID
		stops = dict(zip(ind, stops))

		return stops


## ===================== WORK :



## ==================== TESTS :

def test_001() :
	R = RoutesMeta()
	assert(R.routes)

	print("All bus routes:")
	print(R.routes)

def test_002() :
	S = Stops(RoutesMeta())

	print("Some bus trajectories:")
	for r in list(S.routes.items())[0:10] :
		print(r)

	print("Some platforms:")
	for s in list(S.stops.items())[0:10] :
		print(s)

def test_003() :
	S = StopsKNN(RoutesMeta())

	(lat, lon) = (22.63279, 120.33447)

	print("Finding bus stops closest to (lat, lon) = ({}, {})...".format(lat, lon))

	kS = S.get_nearest_stops((lat, lon), 10)

	for s in json.loads(json.dumps(kS)).values() :
		assert(type(s['distance']) is float)

	for (i, s) in kS.items() :
		print("{}m -- {} (SID: {})".format(int(round(s['distance'])), s['nameZh']['en'], i))

	print("Grouped by name:")
	for (j, s) in reindex_by_key(kS.values(), (lambda s : s['nameZh']['tw'])).items() :
		print(j, s)

def test_004() :
	for i in range(10) :
		print("Executing wget call #{}".format(i+1))
		wget("https://www.google.com/")

def tests() :
	test_004()

## ==================== ENTRY :

if (__name__ == "__main__") :
	tests()

