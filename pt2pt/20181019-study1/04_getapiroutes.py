#!/usr/bin/python3

# RA, 2018-10-22

## ================== IMPORTS :

import commons
import os
import time
import json
import inspect
import random
import urllib.request


## ==================== NOTES :

pass


## ==================== INPUT :

IFILE = {
	'routes-meta-js' : "ORIGINALS/04/kaohsiung_bus_routes.json",
}


## =================== OUTPUT :

OFILE = {
	'BusStop-info-js' : "OUTPUT/04/kaohsiung_bus_routes/route_{route_id}.json",
}

commons.makedirs(OFILE)


## ==================== PARAM :

PARAM = {
	'BusStop-API-URL' : {
		'en' : "http://ibus.tbkc.gov.tw/cms/en/api/route/{route_id}/stop",
		'zh' : "http://ibus.tbkc.gov.tw/cms/api/route/{route_id}/stop",
	}
}


## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))

# Log which files are opened
def logged_open(filename, mode='r', *argv, **kwargs) :
	print("({}):\t{}".format(mode, filename))
	return open(filename, mode, *argv, **kwargs)


## ===================== WORK :

def download() :

	# Load the route meta-info
	with logged_open(IFILE['routes-meta-js'], 'r') as f :
		routes = [ (route['Id'], route) for route in json.load(f)['data']['zh']['route'] ]

	# Check that route IDs are unique
	assert(len(dict(routes)) == len(routes))
	# before converting:
	routes = dict(routes)

	for (route_id, route) in routes.items() :

		print("")
		print("Processing next:", route['NameZh'])

		out = OFILE['BusStop-info-js'].format(route_id=route_id)

		if os.path.isfile(out) :
			print("File {} already exists; skipping download".format(out))
			continue

		urls = PARAM['BusStop-API-URL']

		# JSON structure for this route
		J = { }

		for (lang, url) in urls.items() :

			# See if already downloaded (legacy)...

			old = OFILE['BusStop-info-js'].format(route_id=("{}_{}".format(route_id, lang)))
			if os.path.isfile(old) :

				print("Loading from", old)

				with open(old, 'r') as f :
					j = json.load(f)

			else :

				time.sleep(random.uniform(1, 2))

				url = url.format(route_id=route_id)
				print("Loading from", url)

				with urllib.request.urlopen(url) as response :
					j = json.loads(response.read().decode("utf-8"))

			J[lang] = j

		with logged_open(out, 'w') as f :
			json.dump(J, f)

	print("DONE")


## ==================== ENTRY :

if (__name__ == "__main__") :

	input("Press ENTER to start")
	download()
