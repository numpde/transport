#!/usr/bin/python3

# AUTHOR, DATE

## ================== IMPORTS :

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
	'BusStop-info-js' : "OUTPUT/04/kaohsiung_bus_routes/route_{route_id}_{lang}.json",
}

# Create output directories
for f in OFILE.values() : os.makedirs(os.path.dirname(f), exist_ok=True)


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

		print("Processing next:", route['NameZh'])

		for (lang, url) in PARAM['BusStop-API-URL'].items() :

			url = url.format(route_id=route_id)
			out = OFILE['BusStop-info-js'].format(route_id=route_id, lang=lang)

			if os.path.isfile(out) :

				print("File {} already exists; skipping download".format(out))

			else :

				print("File {} not found".format(out))

				time.sleep(1)
				print("Loading from", url)

				with urllib.request.urlopen(url) as response :
					with logged_open(out, 'wb') as f :
						f.write(response.read())

				time.sleep(random.uniform(2, 4))

		print("")

	print("DONE")

## ==================== ENTRY :

if (__name__ == "__main__") :

	input("Press ENTER to start")
	download()
