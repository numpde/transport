#!/usr/bin/python3

# RA, 2018-11-15

## ================== IMPORTS :

from helpers import commons, maps, graph

from math import sqrt, floor

import re
import json
import random
import inspect
import traceback
import networkx as nx
import numpy as np
import datetime as dt

from sklearn.neighbors import NearestNeighbors

from copy import deepcopy

from itertools import chain, product, groupby

from difflib import SequenceMatcher
from sklearn.cluster import AgglomerativeClustering

from joblib import Parallel, delayed
from progressbar import progressbar

import matplotlib as mpl
# Note: do not import pyplot here -- may need to select renderer


## ==================== NOTES :

pass


## ================= FILE I/O :

open = commons.logged_open


## ==================== INPUT :

IFILE = {
	'mapmatched' : "OUTPUT/14/mapmatched/{scenario}/{routeid}-{direction}/UV/{mapmatch_uuid}.{ext}",
}


## =================== OUTPUT :

OFILE = {
	'mapped_routes' : "OUTPUT/15/mapped/{scenario}/{routeid}-{direction}.{ext}",

	'progress' : "OUTPUT/15/progress/UV/progress_{stage}.{ext}",
}

commons.makedirs(OFILE)


## ==================== PARAM :

PARAM = {
	'mapbox_api_token' : commons.token_for('mapbox'),

	'do_path_alignment' : True,
	'do_path_clustering' : False,

	'quality_min_src/route' : 12,
	'quality_min_wp/src' : 4,

	'candidates_oversampling' : 1/2,
	'candidates_min#' : 12,
	'candidates_max#' : 48,
	'candidates_rounds' : 30,

	'candidates_all_repeat' : 10,

	# Final candidate route score
	'route_fitness' : (lambda m: (m['covr'] + m['miss'] + sqrt(m['dist']) + (m['turn'] / m['dist']))),

	'#jobs' : commons.cpu_frac(0.7),
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
	(k_gp, k_p) = ('geo_path', 'path')
	# The geopath as a list of coordinate pairs
	src[k_gp] = list(map(tuple, src[k_gp]))
	assert(len(src[k_p]) == len(src[k_gp])), "Node path and geo-path do not seem to match."
	# Remove sequentially repeated geo-coordinates
	(src[k_gp], src[k_p]) = map(list, zip(*commons.remove_repeats(list(zip(src[k_gp], src[k_p])), key=(lambda gp_p : gp_p[0]))))
	(list(commons.remove_repeats(src[k_gp])) == list(src[k_gp])) or commons.logger.warning("Repeats in geopath")
	return src


# Divide edges of a path until they are small
def refine_georoute(route, maxlen=5) :
	if (len(route) < 2) :
		return route
	elif (len(route) == 2) :
		if (commons.geodesic(*route) <= maxlen) :
			return route
		else :
			m = tuple(sum(c) / len(c) for c in zip(*route))
			route = [route[0], m, route[1]]
	return commons.remove_repeats(chain.from_iterable(refine_georoute(e) for e in zip(route, route[1:])))


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

def distill_geopath_ver2(sources) :

	# Processing flowchart:
	#
	# geopaths ==> templates --> candidates --> route
	#                 ^==============="

	all_waypoints = set(map(tuple, chain.from_iterable(src['waypoints_used'] for src in sources)))

	geopaths = [tuple(src['geo_path']) for src in sources]

	if (len(geopaths) < 2) :
		raise ValueError("At least two paths are required")

	# Image of provided route variants and the original waypoints
	with open(OFILE['progress'].format(stage='candidates_{round:02d}'.format(round=0), ext='png'), 'wb') as fd :
		maps.write_track_img(waypoints=all_waypoints, tracks=geopaths, fd=fd, mapbox_api_token=PARAM['mapbox_api_token'])

	# Return for each point in 'pp' its minimal distance to the cloud of points 'cloud'
	def dist2closest(pp: list, cloud: set) :
		knn = graph.compute_geo_knn(dict(enumerate(cloud)), leaf_size=20)['knn_tree']
		return [np.min(knn.query(np.asarray(p).reshape(1, -1), k=1)[0]) for p in pp]

	# Returns a dictionary
	# Directed edge --> Counter of possible next points
	def predictor(paths) :
		from collections import defaultdict, Counter
		node_next = defaultdict(Counter)
		for p in paths :
			p = list(p) + [None]
			for (e, c) in zip(zip(p, p[1:]), p[2:]) :
				node_next[e].update([c])
		return dict(node_next)

	# Estimate path tail from an edge using a predictor
	def complete(e, node_next) :
		while node_next.get(e) :
			a = random.choices(list(node_next[e].keys()), weights=list(node_next[e].values()), k=1).pop()
			e = (e[1], a)
			if a : yield a

	#
	def get_candidates_from(templates, rounds=0) :

		# # Show all route candidates in one image
		# with open(OFILE['progress'].format(stage='templates_{round:02d}'.format(round=rounds), ext='png'), 'wb') as fd :
		# 	maps.write_track_img(waypoints=[], tracks=templates, fd=fd, mapbox_api_token=PARAM['mapbox_api_token'])

		if (rounds >= PARAM['candidates_rounds']) :
			return templates

		if (len(set(templates)) < 2) :
			return templates

		# commons.logger.info("Launched candidate round {} with {} templates...".format(rounds, len(templates)))

		node_forw = predictor(templates)
		node_back = predictor(map(reversed, templates))

		# Collect route candidates, some of them multiple times
		candidates = []
		while (len(candidates) < PARAM['candidates_min#']) or (len(candidates) <= len(set(candidates)) * (1 + PARAM['candidates_oversampling'])) :

			if (len(candidates) >= PARAM['candidates_max#']) : break

			# Pick an edge to extend a route in both directions
			root_edge = tuple(random.choice(list(chain.from_iterable(zip(gp, gp[1:]) for gp in templates))))

			# Extended route
			candidate = tuple(reversed(list(complete(tuple(reversed(root_edge)), node_back)))) + root_edge + tuple(complete(root_edge, node_forw))

			# Record candidate
			if (len(candidate) >= 2) :
				candidates.append(candidate)

		assert(len(candidates)), "No route candidates!"

		# Next hierarchy round
		return get_candidates_from(candidates, rounds + 1)

	# COLLECT ALL CANDIDATES REPEATEDLY
	commons.logger.info("Collecting candidates...")
	candidates = list(chain.from_iterable(
		Parallel(n_jobs=PARAM['#jobs'])(
			delayed(get_candidates_from)(geopaths)
			for __ in progressbar(range(PARAM['candidates_all_repeat']))
		)
	))

	# Compute the relative frequency of the candidates, which we interpret as likelihood of being the correct route
	route_freq = {route : (len(list(g)) / len(candidates)) for (route, g) in groupby(sorted(candidates))}

	# Keep only the most frequent candidates
	route_freq = {r : f for (r, f) in route_freq.items() if (f >= min(sorted(route_freq.values(), reverse=True)[0:10]))}


	# Q: individual coverage for each set of waypoints?

	commons.logger.info("Computing metrics...")

	def compute_route_metrics(route) :
		# Subdivide long edges
		fine_route = refine_georoute(route)
		# Collect the metrics
		# commons.logger.debug("Total number of waypoints: {}, length of route candidate: {}".format(len(all_waypoints), len(fine_route)))
		metrics = {
			'path': route,
			# (*): Less is better
			# Coverage of waypoints (*)
			'covr': np.mean(dist2closest(all_waypoints, fine_route)),
			# Deviation from waypoints (*)
			'miss': np.mean(dist2closest(fine_route, all_waypoints)),
			# Total length (*)
			'dist': sum(commons.geodesic(*e) for e in zip(route, route[1:])),
			# Total turns, in degrees (*)
			'turn': sum(abs(graph.angle(p, q, r)) for (p, q, r) in zip(route, route[1:], route[2:])),
		}
		# Final score
		metrics['CALL'] = PARAM['route_fitness'](metrics)

		return metrics


	# COLLECT METRICS FOR EACH CANDIDATE
	commons.logger.info("Computing candidate metrics...")
	candidate_metrics = Parallel(n_jobs=PARAM['#jobs'])(
		delayed(compute_route_metrics)(route)
		for route in progressbar(route_freq)
	)

	# Rekey by route
	candidate_metrics = { m['path']: m for m in candidate_metrics }

	# Additional metrics, in the order of decreasing prior likelihood
	candidate_metrics = {
		route : {'rank': n, 'freq': route_freq[route], **candidate_metrics[route]}
		for (n, route) in enumerate(sorted(route_freq, key=(lambda r : -route_freq[r])))
	}


	def observe(metrics) :
		commons.logger.info("Candidate frequency rank #{}: CALL={}".format(metrics['rank'], metrics['CALL']))
		fn = OFILE['progress'].format(stage='candidate_{rank:04d}'.format(rank=metrics['rank']), ext='png')
		with open(fn, 'wb') as fd :
			maps.write_track_img(waypoints=[], tracks=[metrics['path']], fd=fd, mapbox_api_token=PARAM['mapbox_api_token'])
		return metrics

	#
	commons.logger.info("Candidates summary:")
	for m in sorted(candidate_metrics.values(), key=(lambda m : m['CALL'])) :
		observe(m)

	# Winner candidate
	route = min(candidate_metrics, key=(lambda r : candidate_metrics[r]['CALL']))

	return route


## =================== MASTER :

def map_routes() :

	commons.seed()

	# Dictionary of key-values like
	# ('Kaohsiung/TIME', 'KHH144', '0') --> List of files [PATHTO]/Kaohsiung/TIME/KHH144/0/*.json
	case_files = {
		case : list(g)
		for (case, g) in groupby(
			commons.ls(IFILE['mapmatched'].format(scenario="**", routeid="*", direction="*", mapmatch_uuid="*", ext="json")),
			key=(lambda s : re.fullmatch(IFILE['mapmatched'].format(scenario="(.*)", routeid="([A-Z0-9]+)", direction="([01])", mapmatch_uuid=".*", ext="json"), s).groups())
		)
	}

	# DEBUG
	scenario = "Kaohsiung/20181105-20181111"
	# case = (scenario, 'KHH16', '1')
	# case = (scenario, 'KHH12', '1')
	# case = (scenario, 'KHH100', '0')
	case = (scenario, 'KHH11', '1')
	# case = (scenario, 'KHH116', '0')
	# case = (scenario, 'KHH1221', '0')
	# case = (scenario, 'KHH1221', '1')
	# case = (scenario, 'KHH131', '0')
	case_files = { case : case_files[case] }

	for ((scenario, routeid, dir), files) in case_files.items() :
		commons.logger.info("===")
		commons.logger.info("Mapping route {}, direction {} (from scenario {})...".format(routeid, dir, scenario))

		try :

			if not files :
				commons.logger.warning("No mapmatch files to distill")
				continue

			# Load map-matched variants
			sources = { fn : preprocess_source(commons.zipjson_load(fn)) for fn in files }

			commons.logger.info("Number of sources before quality filter: {}".format(len(sources)))

			# Quality filter
			def is_qualified(src) :
				if (len(src['waypoints_used']) < PARAM['quality_min_wp/src']) : return False
				return True

			# Filter quality
			sources = { fn : src for (fn, src) in sources.items() if is_qualified(src) }

			commons.logger.info("Number of sources: {}".format(len(sources)))

			if (len(sources) < PARAM['quality_min_src/route']) :
				commons.logger.warning("too few sources -- skipping")
				continue

			# Combine map-matched variants
			route = distill_geopath_ver2(sources.values())

			fn = OFILE['mapped_routes'].format(scenario=scenario, routeid=routeid, direction=dir, ext="{ext}")
			commons.makedirs(fn)

			with open(fn.format(ext="json"), 'w') as fd :
				json.dump({
					'geo-path' : route,
					'sources' : sources,
					'datetime' : str(dt.datetime.now().astimezone(tz=None)),
				}, fd)

			with open(fn.format(ext="gpx"), 'w') as fd :
				fd.write(graph.simple_gpx([], [route]).to_xml())

			with open(fn.format(ext="png"), 'wb') as fd :
				maps.write_track_img([], [route], fd, PARAM['mapbox_api_token'])

			with open(fn.format(ext="src.png"), 'wb') as fd :
				maps.write_track_img([], [src['geo_path'] for src in sources.values()], fd, PARAM['mapbox_api_token'])

		except Exception as e :
			commons.logger.warning("Mapping failed ({}). Traceback: \n{}".format(e, traceback.format_exc()))

	commons.logger.info("Done.")


## ==================== ENTRY :

if (__name__ == "__main__") :
	map_routes()
