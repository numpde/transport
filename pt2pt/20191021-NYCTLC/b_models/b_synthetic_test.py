
from a_manhattan_metric import refine_effective_metric, options_refine_effective_metric
from helpers.commons import makedirs, section, parallel_map, this_module_body, range, slice

import os
import json
import pickle

from math import log2
from time import sleep
from glob import glob
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import networkx as nx

from scipy.sparse import dok_matrix

from types import SimpleNamespace
from collections import Counter
from itertools import product
from more_itertools import pairwise, first, last

from contextlib import suppress

import seaborn as sb
import matplotlib as mpl
import matplotlib.pyplot as plt

from percache import Cache
cache = Cache(makedirs("synthetic/UV/percache_runs"), livesync=True)

# ~~~~ LOGGING ~~~~ #

import logging as logger
logger.basicConfig(level=logger.DEBUG, format="%(levelname)-8s [%(asctime)s] @%(funcName)s : %(message)s", datefmt="%Y%m%d %H:%M:%S %Z")
logger.getLogger('matplotlib').setLevel(logger.WARNING)
logger.getLogger('PIL').setLevel(logger.WARNING)


# ~~~~ SETTINGS ~~~~ #

PARAM = {
	'out_experiment_results': os.path.join(os.path.dirname(__file__), "synthetic", "{aliquot}", "results.{ext}"),
	'savefig_args': dict(bbox_inches='tight', pad_inches=0, jpeg_quality=0.9, dpi=300),
}


# ~~~ HELPERS ~~~~ #

class GraphPath:
	def __init__(self, graph, weight="len"):
		self.graph = graph
		self.weight = weight

	def __call__(self, uv):
		self.graph: nx.DiGraph
		path = nx.shortest_path(self.graph, source=uv[0], target=uv[1], weight=self.weight)
		return path


# ~~~~ DATA SOURCE ~~~~ #

def odd_king_graph(xn=8, yn=8, scale=1.0) -> nx.DiGraph:
	"""
	Constructs a Manhattan-like chessboard directed graph.

	Example:
		import networkx as nx
		import matplotlib.pyplot as plt
		g = king_graph(xn=4, yn=8, scale=2)
		nx.draw(g, pos=nx.get_node_attributes(g, "pos"))
		plt.show()
	"""

	g = nx.DiGraph()

	for i in range(xn):
		for (a, b) in pairwise(range(yn)):
			if (i % 2):
				g.add_edge((i, b), (i, a))
			else:
				g.add_edge((i, a), (i, b))

	for j in range(yn):
		for (a, b) in pairwise(range(xn)):
			if (j % 2):
				g.add_edge((a, j), (b, j))
			else:
				g.add_edge((b, j), (a, j))

	nx.set_node_attributes(g, name="pos", values={n: (n[0] * scale, n[1] * scale) for n in g.nodes})
	nx.set_node_attributes(g, name="loc", values={n: (n[1] * scale, n[0] * scale) for n in g.nodes})
	nx.set_edge_attributes(g, name="len", values=scale)

	g = nx.convert_node_labels_to_integers(g)

	return g


# ~~~~ EXPERIMENTS ~~~~ #

@cache
def experiment(graph_size=32, ntrips=1000, noise=0.2, num_rounds=64):

	graph = odd_king_graph(xn=graph_size, yn=graph_size, scale=50)
	logger.debug(F"Constructed 'odd king' graph with {graph.number_of_nodes()} nodes")

	# nodes = pd.DataFrame(data=nx.get_node_attributes(graph, name="loc"), index=["lat", "lon"]).T

	random_state = np.random.RandomState(1)

	secret_met = {e: (v * (1 + noise * random_state.random())) for (e, v) in nx.get_edge_attributes(graph, name="len").items()}
	nx.set_edge_attributes(graph, name="met", values=secret_met)

	assert(sorted(list(graph.nodes)) == sorted(range(graph.number_of_nodes()))), "Expect node labels to be 0, 1, ..."

	random_state = np.random.RandomState(2)

	trips = pd.DataFrame(
		data=((random_state.choice(graph.number_of_nodes(), size=2, replace=False)) for __ in range(ntrips)),
		columns=["u", "v"],
	)

	# logger.warning("Invoking trips.drop_duplicates")
	# trips = trips.drop_duplicates()

	with section(F"Collecting {len(trips)} secret trips", print=logger.debug):
		secret_paths = parallel_map(GraphPath(graph, weight="met"), zip(trips.u, trips.v))

	coverage = pd.Series(dict(Counter(e for path in secret_paths for e in pairwise(path))))

	logger.debug([
		F"{nedges} edges x{cov}"
		for (cov, nedges) in sorted(Counter(coverage).items(), key=first)
	])

	def secret_path_length(path):
		return sum(secret_met[e] for e in pairwise(path))

	trips['trip_distance/m'] = [secret_path_length(path) for path in secret_paths]

	# nx.draw(graph, pos=nx.get_node_attributes(graph, name="pos"))
	# for (__, trip) in trips.iterrows():
	# 	path = nx.shortest_path(graph, source=trip.u, target=trip.v, weight="met")
	# 	plt.plot(nodes.lon[path], nodes.lat[path], 'b-')
	# plt.show()

	history = pd.DataFrame({'secret': secret_met, 0: pd.Series(nx.get_edge_attributes(graph, name="len"))})

	def cb(info):
		if (info.round == (2 ** round(log2(info.round)))):
			history[info.round] = info.edges_met

	opt = options_refine_effective_metric()
	opt.min_trip_distance_m = 0.1
	opt.max_trip_distance_m = 1e8
	opt.num_rounds = num_rounds

	refine_effective_metric(graph, trips, callback=cb, opt=opt)

	return history


def run_experiments() -> pd.DataFrame:
	aliquot = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

	setups = pd.DataFrame(
		# data=list(product([32, 64], [0.1, 0.2, 0.3, 0.4, 0.5], [(2 ** n) for n in range[7, 13]])),
		data=list(product([32, 64], [0.1, 0.2, 0.3, 0.4, 0.5], [(2 ** n) for n in range[7, 10]])),
		# data=list(product([64], [0.5], [2 ** 13])),
		# data=list(product([(2 ** n) for n in range[2, 6]], [0.1, 0.2, 0.4], [100, 1000, 10000])),
		# data=list(product([4, 8, 16], [0.1], [10, 100])),
		columns=["graph_size", "noise", "ntrips"],
	)

	logger.debug('\n'.join(map(str, ["Experiments:", setups])))

	for setup in setups.itertuples(index=False):

		# Preserve datatypes
		setup = dict(setup._asdict())

		# setup = setup.astype({'graph_size': int, 'noise': float, 'ntrips': int})

		with section(F"Experiment {setup} is on", print=logger.info):
			with pd.option_context('mode.chained_assignment', None):
				history = experiment(**setup, num_rounds=64)

		with open(makedirs(PARAM['out_experiment_results'].format(aliquot=aliquot, ext="pkl")), 'ab') as fd:
			pickle.dump({**setup, 'history': history}, fd)

		# results.to_json(makedirs(PARAM['out_experiment_results'].format(aliquot=aliquot, ext="json")))

		with open(makedirs(PARAM['out_experiment_results'].format(aliquot=aliquot, ext="json")), 'w') as fd:
			json.dump(
				{
					'setups': setups.to_json(),
					'script': this_module_body(),
					'timestamp': datetime.now(tz=timezone.utc).isoformat(),
				},
				fd
			)

	return setups


# ~~~~ PLOTS ~~~~ #

def plot_results():
	def summary(history: pd.DataFrame):
		history = history.div(history['secret'], axis=0).drop(columns=['secret', 0])
		history = (100 * history.transform(np.log10).mean(axis=0)).transform(np.abs)
		return history

	cat = pd.DataFrame(glob(PARAM['out_experiment_results'].format(aliquot="*", ext="*")), columns=["file"])
	cat = cat.assign(dir=list(map(os.path.dirname, cat.file)))
	cat = cat.assign(sig=list(map(os.path.basename, cat.dir)))
	cat = cat.assign(ext=list(map(last, map(os.path.splitext, cat.file))))
	cat = cat.pivot(index='dir', columns='ext', values='file')

	for (folder, meta) in cat.iterrows():
		logger.info(F"Folder: {os.path.relpath(folder, os.path.dirname(__file__))}")

		df: pd.DataFrame
		df = pd.DataFrame()
		with open(meta['.pkl'], 'rb') as fd:
			with suppress(EOFError):
				while 1:
					df = df.append(pickle.load(fd), ignore_index=True)

		df = df[df.ntrips > 10]
		df = df[~df.history.isna()]

		df = df.astype({'noise': float, 'graph_size': int, 'ntrips': int})

		for (noise, df0) in df.groupby(df['noise']):
			df0 = df0.copy()
			for (graph_size, df1) in df0.groupby(df0.graph_size):

				image_file = makedirs(os.path.join(folder, "images", F"noise={noise}".replace(".", "p"), F"graph_size={graph_size}", "convergence.{ext}"))

				fig: plt.Figure
				ax1: plt.Axes
				(fig, ax1) = plt.subplots()
				for (ntrips, history) in zip(df1['ntrips'], map(summary, df1['history'])):
					ax1.plot(history, marker='.', ls='--', label=ntrips)
				ax1.set_xscale("log")
				ax1.set_yscale("log")
				ax1.set_ylabel("Geometric average relative error, %")
				# ax1.set_ylim(1e-2, 1e0)
				ax1.grid()
				ax1.set_title(F"Graph size: {graph_size}, noise: {noise}")
				ax1.legend()

				fig.savefig(image_file.format(ext="png"), **PARAM['savefig_args'])

				plt.close(fig)


# ~~~~ ENTRY ~~~~ #

def main():
		with section("Plotting past results", print=logger.info):
			try:
				logger.debug("Press CTRL+C to skip and rerun the experiments instead...")
				sleep(5)
				plot_results()
				return
			except KeyboardInterrupt:
				pass

		with section("Running new set of experiments", print=logger.info):
			sleep(4)
			run_experiments()

if __name__ == '__main__':
	main()
