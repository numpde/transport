#!/usr/bin/python3

# RA, 2018-11-15

## ================== IMPORTS :

from helpers import commons, maps, graph

from math import sqrt, floor

import re
import json
import glob
import random
import inspect
import traceback
import numpy as np
import datetime as dt

from sklearn.neighbors import NearestNeighbors

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

	'progress' : "OUTPUT/15/progress/UV/progress_{stage}.{ext}",
}

commons.makedirs(OFILE)


## ==================== PARAM :

PARAM = {
	'mapbox_api_token' : commons.logged_open(".credentials/UV/mapbox-token.txt", 'r').read(),

	'do_path_alignment' : True,
	'do_path_clustering' : False,

	'quality_min_src/route' : 12,
	'quality_min_wp/src' : 4,

	'candidates_oversampling' : 2/3,
	'candidates_min#' : 24,
}


## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))

# Similarity index of two paths
def sequence_sim(a, b) :
	return SequenceMatcher(None, a, b).ratio()


# Return the most common element of an iterable 'a'
def commonest(a) :
	a = list(a)
	return max(a, key=a.count)


def preprocess_source(src) :
	k = 'geo_path'
	# The geopath as a list of coordinate pairs
	src[k] = list(map(tuple, src[k]))
	(commons.remove_repeats(src[k]) == src[k]) or print("Warning: repeats in geopath.")
	return src


# Divide edges of a path until they are small
def refine_route(route, maxlen=5) :
	if (len(route) < 2) :
		return route
	elif (len(route) == 2) :
		if (commons.geodesic(*route) <= maxlen) :
			return route
		else :
			m = tuple(sum(c) / len(c) for c in zip(*route))
			route = [route[0], m, route[1]]
	return commons.remove_repeats(chain.from_iterable(refine_route(e) for e in zip(route, route[1:])))


def into_two_clusters(geopaths, keep=3/4, seq_sim=sequence_sim) :
	# Number of geopaths
	ngp = len(geopaths)

	# Affinity matrix
	M = np.zeros((ngp, ngp))
	for ((i, gp1), (j, gp2)) in product(enumerate(geopaths), repeat=2):
		M[i, j] = 1 - seq_sim(gp1, gp2)

	# Clustering: we assume that 3/4 of runs are of the same kind
	# with the rest being "random" runs that do not map to the route.
	# That makes at most (1/4 * ngp) clusters.
	labels = list(AgglomerativeClustering(linkage='complete', affinity='precomputed', n_clusters=round((1 - keep) * ngp)).fit_predict(M))

	# Discarded geopaths (outside of the largest cluster)
	discards = [gp for (gp, label) in zip(geopaths, labels) if (label != commonest(labels))]

	# Get the mapmatched paths corresponding to the largest cluster
	retained = [gp for (gp, label) in zip(geopaths, labels) if (label == commonest(labels))]

	return (retained, discards)


## =================== SLAVES :

def distill_geopath_ver1(geopaths) :

	if (len(geopaths) < 2) : raise ValueError("At least two paths are required")

	if PARAM['do_path_clustering'] :
		(geopaths, discards) = into_two_clusters(geopaths, keep=3/4)

		# Image of discarded paths
		maps.write_track_img([], discards, OFILE['progress'].format(stage='cluster-discarded', ext='png'), PARAM['mapbox_api_token'])

	# Image of retained paths
	maps.write_track_img([], geopaths, OFILE['progress'].format(stage='cluster', ext='png'), PARAM['mapbox_api_token'])

	def consensus(geopaths) :
		# Just in case, remove empty paths, make a list
		geopaths = [list(gp) for gp in geopaths if gp]

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

	# Mapping many to one
	route = list(consensus(geopaths))

	return route


def distill_geopath_ver2(sources) :
	geopaths = [src['geo_path'] for src in sources]

	all_waypoints = set(map(tuple, (chain.from_iterable(src['waypoints'] for src in sources))))
	all_nodes_pos = set(map(tuple, (chain.from_iterable(src['geo_path'] for src in sources))))

	if (len(geopaths) < 2) : raise ValueError("At least two paths are required")

	# Image of provided route variants and the original waypoints
	maps.write_track_img(all_waypoints, geopaths, OFILE['progress'].format(stage='templates', ext='png'), PARAM['mapbox_api_token'])

	def dist(p, knn: NearestNeighbors) :
		return np.min(knn.query(np.asarray(p).reshape(1, -1), k=1)[0])

	def dist2closest(pp: list, cloud: set) :
		knn = graph.compute_geo_knn(dict(enumerate(cloud)), leaf_size=3)['knn_tree']
		return [dist(p, knn) for p in pp]

	edge_freq = dict()
	for gp in geopaths :
		for e in zip(gp, gp[1:]) :
			edge_freq[e] = edge_freq.get(e, 0) + 1

	def plotter(fig, ax) :
		for (e, f) in edge_freq.items() :
			(y, x) = zip(*e)
			ax.plot(x, y, 'b-', linewidth=1, c="C{}".format(min(f, 9)))

	maps.write_track_img([], [], OFILE['progress'].format(stage='edge-frequencies', ext='png'), PARAM['mapbox_api_token'], plotter=plotter)

	def count_so_far(lst) :
		counts = dict()
		for i in lst :
			counts[i] = counts.get(i, 0) + 1
			yield counts[i]

	node_next = dict()
	node_prev = dict()
	for gp in geopaths :
		# Append sequential-counter-tag to nodes
		gp = list(zip(gp, count_so_far(gp)))
		for (a, b) in zip(gp, gp[1:]) :
			node_next[a] = node_next.get(a, []) + [b]
			node_prev[b] = node_prev.get(b, []) + [a]

	prev_freq = dict()
	for prev in node_prev.values() :
		for p in prev :
			prev_freq[p] = prev_freq.get(p, 0) + 1

	def complete(a, node_next) :
		while node_next.get(a) :
			if (len(node_next[a]) < 2) : return
			a = random.choice(node_next[a])
			if a : yield a

	# Collect distinct route candidates, some of them multiple times
	routes = []
	while (len(routes) < PARAM['candidates_min#']) or (len(routes) <= len(set(routes)) * (1 + PARAM['candidates_oversampling'])) :
		# Pick a node to extend a route in both directions
		root_node = commons.random_subset(prev_freq.keys(), weights=prev_freq.values(), k=1).pop()
		# Extended route
		candidate = tuple(reversed(list(complete(root_node, node_prev)))) + tuple([root_node]) + tuple(complete(root_node, node_next))
		# Strip sequential-counter-tag from nodes
		candidate = tuple(n for (n, _) in candidate)
		if (len(candidate) < 2) : continue
		# Record candidate
		routes.append(candidate)
		# #
		# print("Progress: {}%".format(min(100, floor(100 * (len(routes) / len(set(routes)) / (1 + oversampling))))))

	assert(len(routes)), "No route candidates!"

	maps.write_track_img([], routes, OFILE['progress'].format(stage='route-candidates', ext='png'), PARAM['mapbox_api_token'])


	# Compute frequencies of the candidates
	route_metrics = {route : (len(list(g)) / len(routes)) for (route, g) in groupby(sorted(routes))}

	# Keep only the most frequent candidates
	route_metrics = {r : f for (r, f) in route_metrics.items() if (f in sorted(route_metrics.values(), reverse=True)[0:10])}

	# Final score
	def route_fitness(metrics) :
		return metrics['covr'] + sqrt(metrics['dist']) + (metrics['turn'] / metrics['dist'])

	# Additional metrics
	for (n, route) in enumerate(sorted(route_metrics, key=(lambda r : -route_metrics[r]))) :
		# Re-record frequency in a dictionary of metrics
		route_metrics[route] = { 'freq' : route_metrics[route] }
		# Subdivide long edges
		fine_route = refine_route(route)
		# Coverage of waypoints (*)
		route_metrics[route]['covr'] = np.mean(dist2closest(all_waypoints, fine_route))
		# Deviation from waypoints (*)
		route_metrics[route]['miss'] = np.mean(dist2closest(fine_route, all_waypoints))
		# Total length (*)
		route_metrics[route]['dist'] = sum(commons.geodesic(*e) for e in zip(route, route[1:]))
		# Total turns, in degrees (*)
		route_metrics[route]['turn'] = sum(abs(graph.angle(p, q, r)) for (p, q, r) in zip(route, route[1:], route[2:]))

		# For each of the above (*): Less is better

		# Fitness = final score
		route_metrics[route]['CALL'] = route_fitness(route_metrics[route])

		print(n, route_metrics[route])

		maps.write_track_img([], [route], OFILE['progress'].format(stage='candidate-by-freq_{}'.format(n), ext='png'), PARAM['mapbox_api_token'])

	#
	route = min(route_metrics.keys(), key=(lambda r : route_metrics[r]['CALL']))

	return route


## =================== MASTER :

def map_routes() :

	commons.seed()

	# Dictionary of key-values like
	# 'KHH144-0' --> List of files [PATHTO]/KHH144/0/*.json
	case_files = {
		case : list(g)
		for (case, g) in groupby(
			sorted(list(glob.glob(
				IFILE['mapmatched'].format(routeid="*", direction="*", mapmatch_uuid="*", ext="json")
			))),
			key=(lambda s : re.match(IFILE['mapmatched'].format(routeid="([A-Z0-9]+)", direction="([01])", mapmatch_uuid=".*", ext="json"), s).groups())
		)
	}

	# DEBUG
	# case = ('KHH16', '1')
	# case = ('KHH12', '1')
	# case = ('KHH100', '0')
	# case = ('KHH11', '1')
	# case = ('KHH116', '0')
	case = ('KHH1221', '0')
	case_files = { case : case_files[case] }

	for ((routeid, dir), files) in case_files.items() :

		print("Mapping case {}-{}...".format(routeid, dir))

		try :

			if not files :
				print("Warning: No mapmatch files to distill.")
				continue

			# Load map-matched variants
			sources = { fn : preprocess_source(commons.zipjson_load(fn)) for fn in files }

			print("Number of sources before quality filter: {}".format(len(sources)))

			# Quality filter
			def is_qualified(src) :
				if (len(src['waypoints']) < PARAM['quality_min_wp/src']) : return False
				return True

			# Filter quality
			sources = { fn : src for (fn, src) in sources.items() if is_qualified(src) }

			print("Number of sources: {}".format(len(sources)))

			if (len(sources) < PARAM['quality_min_src/route']) :
				print("Warning: too few sources -- skipping.")
				continue

			# Combine map-matched variants
			route = distill_geopath_ver2(sources.values())

			fn = OFILE['mapped_routes'].format(routeid=routeid, direction=dir, ext="{ext}")
			commons.makedirs(fn)

			with commons.logged_open(fn.format(ext="json"), 'w') as fd :
				json.dump({
					'geo-path' : route,
					'sources' : sources,
					'datetime' : str(dt.datetime.now().astimezone(tz=None)),
				}, fd)

			with commons.logged_open(fn.format(ext="gpx"), 'w') as fd :
				fd.write(graph.simple_gpx([], [route]).to_xml())

			with commons.logged_open(fn.format(ext="png"), 'wb') as fd :
				maps.write_track_img([], [route], fd, PARAM['mapbox_api_token'])

			with commons.logged_open(fn.format(ext="src.png"), 'wb') as fd :
				maps.write_track_img([], [src['geo_path'] for src in sources.values()], fd, PARAM['mapbox_api_token'])

		except Exception as e :
			print("Warning: Mapping failed ({}).".format(e))
			print(traceback.format_exc())

	print("Done.")


## ==================== ENTRY :

if (__name__ == "__main__") :
	map_routes()
