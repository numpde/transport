
# RA, 2018-10-19

## ================== IMPORTS :

import os
import inspect
import datetime
import urllib.request


## ==================== INPUT :

pass


## ==================== PARAM :

PARAM = {
	
	# Regions to download using bounding box [left, bottom, right, top]
	# https://wiki.openstreetmap.org/wiki/API_v0.6

	'regions' : {
		'kaohsiung' : [120.2593, 22.5828, 120.3935, 22.6886],
	},

	# Original API URL, has a limit on data size
	'API-URL2' : "https://api.openstreetmap.org/api/0.6/map?bbox={bbox}",
	# Mirror URL
	'API-URL' : "https://overpass-api.de/api/map?bbox={bbox}",
}


## =================== OUTPUT :

OFILE = {
	# Put the downloaded *.osm files here
	'OSM' : "OUTPUT/00/{region}.osm",
	'OSM-meta' : "OUTPUT/00/{region}_meta.osm",
}

# Create output directories
for f in OFILE.values() : os.makedirs(os.path.dirname(f), exist_ok=True)


## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))

# Log which files are opened
def logged_open(filename, mode='r', *argv, **kwargs) :
	print("({}):\t{}".format(mode, filename))
	return open(filename, mode, *argv, **kwargs)


## ===================== WORK :

def download() :
	for (r, bb) in PARAM['regions'].items() :

		print("Retrieving region:", r)

		url = PARAM['API-URL'].format(bbox=("{0},{1},{2},{3}".format(*bb)))

		with urllib.request.urlopen(url) as response :
			with logged_open(OFILE['OSM'].format(region=r), 'wb') as f :
				f.write(response.read())

		with logged_open(OFILE['OSM-meta'].format(region=r), 'w') as f :
			print("# Bounding box:", file=f)
			print("l={}".format(bb[0]), file=f)
			print("b={}".format(bb[1]), file=f)
			print("r={}".format(bb[2]), file=f)
			print("t={}".format(bb[3]), file=f)
			print("# Source:", file=f)
			print("URL=" + url, file=f)
			print("# Retrieval time:", file=f)
			print("UTC={}".format(datetime.datetime.utcnow().isoformat()), file=f)


## ==================== ENTRY :

if (__name__ == "__main__") :
	download()

