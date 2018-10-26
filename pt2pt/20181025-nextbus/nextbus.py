#!/usr/bin/python3

# RA, 2018-10-26

## ================== IMPORTS :

import os
import builtins
import json
import inspect
import glob
import urllib.request
from collections import defaultdict

## ==================== NOTES :

pass


## ==================== INPUT :

IFILE = {
	'routes' : "request_cache/routes_{lang}.json",
	'stops' : "request_cache/stops_{ID}-{Dir}_{lang}.json",
}


## =================== OUTPUT :

OFILE = {
	'routes' : IFILE['routes'],
	'stops' : IFILE['stops'],

	#'stops-json' : "OUTPUT/05/kaohsiung_bus_stops.json",
}

# Create output directories
#for f in OFILE.values() : os.makedirs(os.path.dirname(f), exist_ok=True)


## ==================== PARAM :

PARAM = {
	'logged-open' : True,

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


#
def wget(url, filename) :
	with urllib.request.urlopen(url) as response :
		with open(filename, 'wb') as f :
			f.write(response.read())

# "Transpose" JSON, rekey by id_key
def lang_reform(I, id_key) :

	J = defaultdict(lambda: defaultdict(dict))

	for (lang, E) in I.items():
		if type(E) is dict :
			E = E.values()
		for e in E :
			for (k, v) in e.items():
				J[e[id_key]][k][lang] = v

	for (i, e) in J.items():
		for (k, V) in e.items():
			V = V.values()
			if (len(set(V)) == 1):
				J[i][k] = set(V).pop()

	# Convert all defaultdict to dict
	J = json.loads(json.dumps(J))

	return J

## ================== CLASSES :

class Routes :

	def routes_init_original(self, load_from_local=False) :

		self.routes_by_lang = { lang : None for lang in PARAM['url-routes'].keys() }

		if load_from_local :

			try :

				for lang in self.routes_by_lang.keys() :
					with open(IFILE['routes'].format(lang=lang), 'r') as f :
						self.routes_by_lang[lang] = json.load(f)

				return True

			except FileNotFoundError as e :

				return False

		else :

			for (lang, url) in PARAM['url-routes'].items() :
				wget(url, OFILE['routes'].format(lang=lang))

			return self.routes_init_original(load_from_local=True)

	def routes_init(self) :

		if not (self.routes_init_original(PARAM['try-local-routes']) or self.routes_init_original()) :
			raise RuntimeError("Failed to load routes")

		self.routes = lang_reform(self.routes_by_lang, 'ID')

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
		self.routes = None
		self.routes_init()


class Stops :

	def init(self) :

		# self.routes[route_id][direction] is a list of stop SIDs
		self.routes = defaultdict(dict)
		# self.stops is a dict of all stops/platforms keyed by SID
		self.stops = defaultdict(dict)

		for (lang, preurl) in PARAM['url-routestops'].items():
			for (i, route) in self.R.routes.items() :

				nroutes = int(route['routes'])
				assert(nroutes in [1, 2])

				for Dir in [1, 2][:nroutes] :

					url = preurl.format(ID=i, Dir=Dir)
					filename = OFILE['stops'].format(ID=i, Dir=Dir, lang=lang)

					if (not os.path.isfile(filename)) or (not PARAM['try-local-routestops']) :
						wget(url, filename)

					# As of 2018-10-27, the following routes get an error response:
					# 1602-1, 2173-2, 2482-1, 371-1

					# Special cases:
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
					self.routes[i][Dir] = []

					for stop in sorted(J, key=(lambda r : int(r['seqNo']))) :

						n = stop['SID']

						self.routes[i][Dir].append(n)

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

							# ETC... WHATEVER, WE SIMPLY DELETE THE 'Id' FIELD

							del stop['Id']


						if n in self.stops[lang] :
							if not (self.stops[lang][n] == stop) :
								print("Ex. A:", self.stops[lang][n])
								print("Ex. B:", stop)
								raise RuntimeError("Data inconsistency")
						else :
							self.stops[lang][n] = stop

		self.stops = lang_reform(self.stops, 'SID')



	def __init__(self, R) :
		assert(type(R) is Routes), "Type checking failed"

		self.R = R
		self.init()

## ===================== WORK :



## ==================== TESTS :

def test_001() :
	R = Routes()
	assert(R.routes)
	print(R.routes)

def test_002() :
	R = Routes()
	S = Stops(R)

def tests() :
	test_002()

## ==================== ENTRY :

if (__name__ == "__main__") :
	tests()

