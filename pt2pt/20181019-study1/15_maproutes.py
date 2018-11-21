#!/usr/bin/python3

# RA, 2018-11-15

## ================== IMPORTS :

from helpers import commons, maps, graph

import re
import json
import glob
import inspect
import traceback
import numpy as np

from itertools import chain, product, groupby

from difflib import SequenceMatcher
from sklearn.cluster import AgglomerativeClustering

import matplotlib as mpl
# Note: do not import pyplot here -- may need to select renderer


## ==================== NOTES :

pass


## ==================== INPUT :

IFILE = {
	'mapmatched' : "OUTPUT/14/mapmatched/{routeid}/{direction}/UV/{mapmatch_uuid}.{ext}",
}


## =================== OUTPUT :

OFILE = {
	'mapped_routes' : "OUTPUT/15/mapped/{routeid}-{direction}.{ext}",
}

commons.makedirs(OFILE)


## ==================== PARAM :

PARAM = {
	'mapbox_api_token' : open(".credentials/UV/mapbox-token.txt", 'r').read(),
}


## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))

# Similarity index of two paths
def pathsim(a, b) :
	return SequenceMatcher(None, a, b).ratio()


def write_track_gpx(waypoints, route, fd) :
	fd.write(graph.simple_gpx(waypoints, [route]).to_xml())

def write_track_img(waypoints, route, fd) :
	mpl.use('Agg')
	import matplotlib.pyplot as plt

	if waypoints : raise NotImplementedError("Writing waypoints not implemented.")

	ax : plt.Axes
	fig : plt.Figure
	(fig, ax) = plt.subplots()
	(y, x) = zip(*route)
	ax.plot(x, y, 'b-', linewidth=2)
	ax.plot(x[0], y[0], 'o', c='g', markersize=3)
	ax.plot(x[-1], y[-1], 'o', c='r', markersize=3)
	axis = commons.niceaxis(ax.axis(), expand=1.1)
	[i.set_fontsize(8) for i in ax.get_xticklabels() + ax.get_yticklabels()]
	ax.axis(axis)
	ax.imshow(maps.get_map_by_bbox(maps.ax2mb(*axis), token=PARAM['mapbox_api_token']), extent=axis, interpolation='quadric', zorder=-100)

	fig.savefig(fd, dpi=180, bbox_inches='tight', pad_inches=0)
	plt.close(fig)


## ===================== WORK :

def distill_geopath(geopaths) :

	def commonest(xx) :
		xx = list(xx)
		return max(xx, key=xx.count)

	# A geopath shall be a list of non-repeating coordinate-tuples
	geopaths = [list(commons.remove_repeats(map(tuple, gp))) for gp in geopaths]

	# Number of map-matched variants
	ngp = len(geopaths)

	# Affinity matrix
	M = np.zeros((ngp, ngp))
	for ((i, gp1), (j, gp2)) in product(enumerate(geopaths), repeat=2) :
		M[i, j] = 1 - pathsim(gp1, gp2)

	# Clustering: we assume that 3/4 of runs are of the same kind
	# with the rest being "random" runs that do not map to the route.
	# That makes at most (1/4 * ngp) clusters.
	labels = list(AgglomerativeClustering(linkage='complete', affinity='precomputed', n_clusters=round(1/4 * ngp)).fit_predict(M))

	# Get the mapmatched paths corresponding to the largest cluster
	geopaths = [gp for (gp, label) in zip(geopaths, labels) if (label == commonest(labels))]

	def consensus(geopaths) :
		# Just in case, remove empty paths, make a list
		geopaths = [gp for gp in geopaths if gp]

		# Define start and end point for the route through consensus
		(p0, p1) = (commonest([gp[0] for gp in geopaths]), commonest([gp[-1] for gp in geopaths]))

		while (len(geopaths) > 1) :

			# The most frequent candidate-location
			p = commonest(gp[0] for gp in geopaths)

			# Where does it appear in other paths?
			for (i, gp) in enumerate(geopaths) :
				if p in gp :
					geopaths[i] = geopaths[i][(gp.index(p) + 1):]

			# Remove emptied geopaths
			geopaths = [gp for gp in geopaths if gp]

			# The start of the route
			if (p == p0) : p0 = None

			# Have met the start, but not past the end of route
			if not p0 : yield p

			# Endpoint
			if (p == p1) : break

	route = list(consensus(geopaths))

	return route


def map_routes() :

	commons.seed()

	case_files = {
		case : list(g)
		for (case, g) in groupby(
			sorted(list(glob.glob(
				IFILE['mapmatched'].format(routeid="*", direction="*", mapmatch_uuid="*", ext="json")
			))),
			key=(lambda s : re.match(IFILE['mapmatched'].format(routeid="([A-Z0-9]+)", direction="([01])", mapmatch_uuid=".*", ext="json"), s).groups())
		)
	}

	for ((routeid, dir), files) in case_files.items() :

		print("Mapping case {}-{}...".format(routeid, dir))

		try :

			if not files :
				print("No mapmatch files to distill.")
				continue

			# Combine map-matched variants
			route = distill_geopath(commons.zipjson_load(fn).get('geo_path') for fn in files)

			fn = OFILE['mapped_routes'].format(routeid=routeid, direction=dir, ext="{ext}")
			commons.makedirs(fn)

			with commons.logged_open(fn.format(ext="json"), 'w') as fd :
				json.dump({ 'geo-path' : route }, fd)

			with commons.logged_open(fn.format(ext="gpx"), 'w') as fd :
				write_track_gpx([], route, fd)

			with commons.logged_open(fn.format(ext="png"), 'wb') as fd :
				write_track_img([], route, fd)

		except Exception as e :
			print("Mapping failed ({}).".format(e))
			print(traceback.format_exc())


## ==================== ENTRY :

if (__name__ == "__main__") :
	map_routes()
