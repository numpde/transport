
# RA, 2019-10-31

# DRAFT


from helpers import maps
from helpers.commons import myname, makedirs, parallel_map

import os
import math
import json
import pickle

import numpy as np
import pandas as pd
import networkx as nx

from glob import glob

from itertools import groupby
from more_itertools import first, last

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from contextlib import contextmanager






# ~~~~ LOGGING ~~~~ #

import logging as logger
logger.basicConfig(level=logger.DEBUG, format="%(levelname)-8s [%(asctime)s] @%(funcName)s : %(message)s", datefmt="%Y%m%d %H:%M:%S %Z")
logger.getLogger('matplotlib').setLevel(logger.WARNING)
logger.getLogger('PIL').setLevel(logger.WARNING)


# ~~~~ NOTATION ~~~~ #

def datapath(fn):
	return os.path.join(os.path.dirname(__file__), "../data_preparation/data/", fn)


# ~~~~ SETTINGS ~~~~ #

PARAM = {
	'road_graph': datapath("road_graph/UV/nx_digraph_naive.pkl"),

	'hourly_graph_metric': "*/{table_name}/{weekday}/{hour}/edges_met.{ext}",
}


# ~~~~ HELPERS ~~~~ #

def load(fn):
	if fn.lower().endswith(".json"):
		try:
			with open(fn, 'r') as fd:
				return json.load(fd)
		except (json.decoder.JSONDecodeError, TypeError):
			return np.nan

	if fn.lower().endswith(".pkl"):
		with open(fn, 'rb') as fd:
			return pickle.load(fd)


# ~~~~ DATA SOURCE ~~~~ #

def get_road_graph() -> nx.DiGraph:
	logger.debug("Loading the initial road graph")
	g = pickle.load(open(PARAM['road_graph'], 'rb'))
	g = g.subgraph(max(nx.strongly_connected_components(g), key=len)).copy()
	return g




table_name = "yellow_tripdata_2016-05"

files = pd.DataFrame(
	data=[
		(os.path.dirname(file), last(os.path.splitext(file)), file)
		for file in glob(PARAM['hourly_graph_metric'].format(table_name=table_name, weekday="*", hour="*", ext="*"))
	],
	columns=["path", "ext", "file"]
).pivot(
	index="path", columns="ext", values="file"
)

files = files[['.json', '.pkl']].sort_index()

files['wh'] = list(map(load, files['.json']))
files = files.dropna(axis=0)

files['wh'] = [(about['weekday'], about['hour']) for about in files['wh']]


df: pd.DataFrame
df = pd.DataFrame(data=dict(zip(files['wh'], map(load, files['.pkl']))))

df = np.maximum(0, df).sum(axis=0) * 1e-3
df.plot(marker='.', ls='--')
plt.show()
exit(9)

# df = np.log2(df)
# df = df.sub(df.quantile(0.1, axis=1), axis=0)
# df = 2 ** df

# print(df); exit()
# df = df.mean(axis=0)
#
# import matplotlib.pyplot as plt
# df.plot(marker='.', ls='--')
# plt.show()


@contextmanager
def nx_draw(graph, nodes=None, edges=None):
	# mpl.use("Agg")

	logger.debug("Preparing to draw graph")

	if nodes is None:
		nodes = pd.DataFrame(data=nx.get_node_attributes(graph, name="loc"), index=["lat", "lon"]).T

	if edges is None:
		edges = pd.DataFrame([
			pd.Series(data=nx.get_edge_attributes(graph, name="met"), name="met"),
			pd.Series(data=nx.get_edge_attributes(graph, name="len"), name="len"),
		]).T
		edges['color'] = (edges.met / edges.len)

	cmap = LinearSegmentedColormap.from_list(name="noname", colors=["g", "y", "r", "brown"])

	# Axes window
	# extent = np.dot(
	# 	[[min(nodes.lon), max(nodes.lon)], [min(nodes.lat), max(nodes.lat)]],
	# 	(lambda s: np.asarray([[1 + s, -s], [-s, 1 + s]]))(0.01)
	# ).flatten()
	# Manhattan south
	extent = [-74, -73.96, 40.73, 40.78]

	logger.debug("Getting the background OSM map")
	osmap = maps.get_map_by_bbox(maps.ax2mb(*extent), style=maps.MapBoxStyle.streets)

	fig: plt.Figure
	ax1: plt.Axes
	(fig, ax1) = plt.subplots()

	logger.debug("Drawing image")

	# The background map
	ax1.imshow(osmap, extent=extent, interpolation='quadric', zorder=-100)

	ax1.axis("off")

	edges.color[edges.color > 1.5] = 1.5
	edges.color[edges.color < 0.9] = 0.9

	g = nx.draw_networkx_edges(
		graph,
		ax=ax1,
		pos=nx.get_node_attributes(graph, name="pos"),
		edge_list=list(edges.index),
		edge_color=list(edges.color),
		edge_cmap=cmap,
		with_labels=False, arrows=False, node_size=0, alpha=1, width=2
	)

	# https://stackoverflow.com/questions/26739248/how-to-add-a-simple-colorbar-to-a-network-graph-plot-in-python
	(vmin, vmax) = (np.min(edges.color), np.max(edges.color))
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
	sm._A = []
	plt.colorbar(sm)

	# https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_directed.html
	# pc = mpl.collections.PatchCollection(g, cmap=cmap)
	# pc.set_array(list(edges.color))
	# plt.colorbar(pc)

	# cax = fig.add_axes([ax1.get_position().x1 + 0.01, ax1.get_position().y0, 0.02, ax1.get_position().height])
	# cbar = fig.colorbar(im, cax=cax)

	ax1.set_xlim(extent[0:2])
	ax1.set_ylim(extent[2:4])

	logger.debug("OK")

	try:
		yield (fig, ax1)
	finally:
		plt.close(fig)




df: pd.DataFrame
df = pd.DataFrame(data=dict(zip(files['wh'], map(load, files['.pkl']))))

graph = get_road_graph()
nx.set_edge_attributes(graph, values=df[(0, 8)], name="met")
# edges = pd.DataFrame({'color': (df[(0, 8)])})

with nx_draw(graph, edges=None):
	plt.show()

