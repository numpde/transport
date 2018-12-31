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

from itertools import chain, product, groupby

from difflib import SequenceMatcher
from sklearn.cluster import AgglomerativeClustering

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
	'mapbox_api_token' : open(".credentials/UV/mapbox-token.txt", 'r').read(),

	'do_path_alignment' : True,
	'do_path_clustering' : False,

	'quality_min_src/route' : 12,
	'quality_min_wp/src' : 4,

	'candidates_oversampling' : 1/2,
	'candidates_min#' : 24,
	'candidates_max#' : 100,

	# Final candidate route score
	'route_fitness' : (lambda m: (m['covr'] + m['miss'] + sqrt(m['dist']) + (m['turn'] / m['dist']))),
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
	(list(commons.remove_repeats(src[k_gp])) == list(src[k_gp])) or print("Warning: repeats in geopath.")
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
	all_waypoints = set(map(tuple, (chain.from_iterable(src['waypoints_used'] for src in sources))))

	geopaths = [src['geo_path'] for src in sources]

	if (len(geopaths) < 2) :
		raise ValueError("At least two paths are required")


	# Image of provided route variants and the original waypoints
	with open(OFILE['progress'].format(stage='templates', ext='png'), 'wb') as fd :
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

	node_forw = predictor(geopaths)
	node_back = predictor(map(reversed, geopaths))


	# g = nx.DiGraph()
	# p = (22.6211862, 120.3456268)
	# #
	# def populate_from(P, depth) :
	# 	g.add_node(P)
	# 	g.nodes[P]['pos'] = (P[0][1], P[0][0]) # Geo-location
	# 	g.nodes[P]['win'] = P[1] # Winding
	#
	# 	if (depth <= 0) : return
	#
	# 	target = list(node_forw.get(P, []))
	# 	for Q in set(target) :
	# 		g.add_edge(P, Q, flow=(target.count(Q) / len(target)))
	# 		populate_from(Q, depth - 1)
	# #
	# populate_from((p, 1), depth=15)
	#
	# import matplotlib.pyplot as plt
	# def plotter(fig, ax: plt.Axes) :
	# 	ax.tick_params(axis='both', which='both', labelsize='xx-small')
	# 	# Removing axis: https://stackoverflow.com/a/26610602/3609568
	# 	ax.axis('off')
	# 	ax.get_xaxis().set_visible(False)
	# 	ax.get_yaxis().set_visible(False)
	# 	#nx.draw_networkx_nodes(g, dict(g.nodes.data('pos')), ax=ax, node_size=0.1, node_shape='x')
	# 	for (a, b, flow) in g.edges.data('flow') :
	# 		nx.draw_networkx_edges(g, dict(g.nodes.data('pos')), edgelist=[(a, b)], ax=ax, node_size=0, width=0.3, edge_color='r', alpha=flow, arrowsize=1)
	# 	#ax.scatter(*zip(*dict(g.nodes.data('pos')).values()), s=0.01, c='r', marker='.', zorder=100)
	# 	ax.scatter(p[1], p[0], s=0.1, c='g', marker='.', zorder=1000)
	#
	# maps.write_track_img([], [], OFILE['progress'].format(stage='debug', ext='png'), PARAM['mapbox_api_token'], plotter=plotter, dpi=1200)
	# exit(39)


	# print("1-next:", node_forw[((22.6211862, 120.3456268), 1)])
	# print("1-prev:", node_prev[((22.6211862, 120.3456268), 1)])
	# print("2-next:", node_forw[((22.6211862, 120.3456268), 2)])
	# print("2-prev:", node_prev[((22.6211862, 120.3456268), 2)])
	#
	# exit(30)

	# Complete path tail from an edge
	def complete(e, node_next) :
		while node_next.get(e) :
			a = random.choices(list(node_next[e].keys()), weights=list(node_next[e].values()), k=1).pop()
			e = (e[1], a)
			if a : yield a

	print("Computing candidates...")

	# Collect distinct route candidates, some of them multiple times
	routes = []
	while (len(routes) < PARAM['candidates_min#']) or (len(routes) <= len(set(routes)) * (1 + PARAM['candidates_oversampling'])) :
		if (len(routes) >= PARAM['candidates_max#']) : break

		# Pick an edge to extend a route in both directions
		root_edge = tuple(random.choice(list(chain.from_iterable(zip(gp, gp[1:]) for gp in geopaths))))
		# Extended route
		candidate = tuple(reversed(list(complete(tuple(reversed(root_edge)), node_back)))) + root_edge + tuple(complete(root_edge, node_forw))
		if (len(candidate) < 2) : continue
		# Record candidate
		routes.append(candidate)
		# #
		# print("Progress: {}%".format(min(100, floor(100 * (len(routes) / len(set(routes)) / (1 + PARAM['candidates_oversampling']))))))
		#

	assert(len(routes)), "No route candidates!"

	# Show all route candidates in one image
	with open(OFILE['progress'].format(stage='route-candidates', ext='png'), 'wb') as fd :
		maps.write_track_img(waypoints=[], tracks=routes, fd=fd, mapbox_api_token=PARAM['mapbox_api_token'])

	# Compute the relative frequency of the candidates, which we interpret as likelihood of being the correct route
	route_freq = {route : (len(list(g)) / len(routes)) for (route, g) in groupby(sorted(routes))}

	# Keep only the most frequent candidates
	route_freq = {r : f for (r, f) in route_freq.items() if (f >= min(sorted(route_freq.values(), reverse=True)[0:10]))}

	# Q: individual coverage for each set of waypoints?

	print("Computing metrics...")

	def compute_route_metrics(route) :
		# Subdivide long edges
		fine_route = refine_georoute(route)
		# Collect the metrics
		print("Total number of waypoints: {}, length of route candidate: {}".format(len(all_waypoints), len(fine_route)))
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

	def notification_filter(metrics) :
		print("Candidate frequency rank #{}: CALL={}".format(metrics['rank'], metrics['CALL']))
		with open(OFILE['progress'].format(stage='candidate-by-rank{}'.format(metrics['rank']), ext='png'), 'wb') as fd :
			maps.write_track_img(waypoints=[], tracks=[metrics['path']], fd=fd, mapbox_api_token=PARAM['mapbox_api_token'])
		return metrics

	# Additional metrics, in the order of decreasing likelihood
	routes_metrics = {
		route : notification_filter({'rank': n, 'freq': route_freq[route], **compute_route_metrics(route)})
		for (n, route) in enumerate(sorted(route_freq, key=(lambda r : -route_freq[r])))
	}

	# Winner candidate
	route = min(routes_metrics, key=(lambda r : routes_metrics[r]['CALL']))

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
	# case = (scenario, 'KHH11', '1')
	# case = (scenario, 'KHH116', '0')
	# case = (scenario, 'KHH1221', '0')
	# case = (scenario, 'KHH1221', '1')
	# case = (scenario, 'KHH131', '0')
	# case_files = { case : case_files[case] }

	for ((scenario, routeid, dir), files) in case_files.items() :
		print("===")
		print("Mapping route {}, direction {} (from scenario {})...".format(routeid, dir, scenario))

		try :

			if not files :
				print("Warning: No mapmatch files to distill.")
				continue

			# Load map-matched variants
			sources = { fn : preprocess_source(commons.zipjson_load(fn)) for fn in files }

			print("Number of sources before quality filter: {}".format(len(sources)))

			# Quality filter
			def is_qualified(src) :
				if (len(src['waypoints_used']) < PARAM['quality_min_wp/src']) : return False
				return True

			# Filter quality
			sources = { fn : src for (fn, src) in sources.items() if is_qualified(src) }

			print("Number of sources: {}".format(len(sources)))

			if (len(sources) < PARAM['quality_min_src/route']) :
				print("Warning: too few sources -- skipping.")
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
			print("Warning: Mapping failed ({}).".format(e))
			print(traceback.format_exc())

	print("Done.")


## ==================== ENTRY :

if (__name__ == "__main__") :
	map_routes()
