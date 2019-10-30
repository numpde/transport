
# RA, 2019-10-29


from helpers import maps
from helpers.commons import myname, makedirs, parallel_map

import os
import math
import json
import pickle
import sqlite3

import numpy as np
import pandas as pd
import networkx as nx

from types import SimpleNamespace

from itertools import product, chain
from more_itertools import pairwise

from contextlib import contextmanager

from sklearn.neighbors import BallTree as Neighbors

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ~~~~ LOGGING ~~~~ #

import logging as logger
logger.basicConfig(level=logger.DEBUG, format="%(levelname)-8s [%(asctime)s] @%(funcName)s : %(message)s", datefmt="%Y%m%d %H:%M:%S %Z")
logger.getLogger('matplotlib').setLevel(logger.WARNING)
logger.getLogger('PIL').setLevel(logger.WARNING)


# ~~~~ NOTATION ~~~~ #

def datapath(fn):
	return os.path.join(os.path.dirname(__file__), "../a_prepare_data/data/", fn)


# ~~~~ SETTINGS ~~~~ #

PARAM = {
	'taxidata': datapath("taxidata/sqlite/UV/db.db"),
	'road_graph': datapath("road_graph/UV/nx_digraph_naive.pkl"),

	# Tolerance for accepting a nearest node (distance in meters)
	'tol_nearest_nodes/m': 20,

	'out_manhattan_metric_path': makedirs(os.path.join(os.path.dirname(__file__), "manhattan_metric/")),
	'savefig_args': dict(bbox_inches='tight', pad_inches=0, jpeg_quality=0.9, dpi=300),
}


# ~~~~ HELPERS ~~~~ #

class NearestNode:
	RAD_PER_DEGREE = math.pi / 180
	EARTH_RADIUS_METERS = 6367.5 * 1e3

	def __init__(self, graph):
		# Point array
		self.X = pd.DataFrame(data=nx.get_node_attributes(graph, "loc")).T
		# Nearest neighbors tree
		self.T = Neighbors(self.X.values * self.RAD_PER_DEGREE, metric="haversine")

	def __call__(self, loc):
		# Get nearest nodes: distance to X and index in X
		(d, i) = np.squeeze(self.T.query(np.asarray(loc) * self.RAD_PER_DEGREE, k=1, return_distance=True))
		# Note: do not sort the Series
		s = pd.Series(index=(self.X.index[list(map(int, i))]), data=(d * self.EARTH_RADIUS_METERS))
		return s


class GraphPathDist:
	def __init__(self, graph, weight="len"):
		self.graph = graph
		self.weight = weight

	def __call__(self, uv):
		self.graph: nx.DiGraph
		path = nx.shortest_path(self.graph, source=uv[0], target=uv[1], weight=self.weight)
		dist = nx.shortest_path_length(self.graph, source=uv[0], target=uv[1], weight=self.weight)
		return (path, dist)


# ~~~~ GRAPHICS ~~~~ #

@contextmanager
def nx_draw_met_by_len(graph, nodes, edges):
	mpl.use("Agg")

	logger.debug("Preparing to draw graph")

	# Backup:
	#
	# nodes = pd.DataFrame(data=nx.get_node_attributes(graph, name="loc"), index=["lat", "lon"]).T
	#
	# edges = pd.DataFrame([
	# 	pd.Series(data=nx.get_edge_attributes(graph, name="met"), name="met"),
	# 	pd.Series(data=nx.get_edge_attributes(graph, name="len"), name="len"),
	# ])

	cmap = LinearSegmentedColormap.from_list(name="noname", colors=["g", "y", "r"])

	# Axes window
	extent = np.dot(
		[[min(nodes.lon), max(nodes.lon)], [min(nodes.lat), max(nodes.lat)]],
		(lambda s: np.asarray([[1 + s, -s], [-s, 1 + s]]))(0.01)
	).flatten()

	logger.debug("Getting the background OSM map")
	osmap = maps.get_map_by_bbox(maps.ax2mb(*extent))

	fig: plt.Figure
	ax1: plt.Axes
	(fig, ax1) = plt.subplots()

	logger.debug("Drawing image")

	# The background map
	ax1.imshow(osmap, extent=extent, interpolation='quadric', zorder=-100)

	ax1.axis("off")

	nx.draw(
		graph,
		ax=ax1,
		pos=nx.get_node_attributes(graph, name="pos"),
		edges=list(edges.index),
		edge_color=list(np.log(edges.met / edges.len)),
		edge_cmap=cmap,
		with_labels=False, arrows=False, node_size=0, alpha=1, width=0.4
	)

	ax1.set_xlim(extent[0:2])
	ax1.set_ylim(extent[2:4])

	try:
		yield (fig, ax1)
	finally:
		plt.close(fig)


# ~~~~ DATA SOURCE ~~~~ #

def get_road_graph() -> nx.DiGraph:
	logger.debug("Loading the initial road graph")
	g = pickle.load(open(PARAM['road_graph'], 'rb'))
	g = g.subgraph(max(nx.strongly_connected_components(g), key=len)).copy()
	return g


def get_trips(table_name, graph, where="") -> pd.DataFrame:
	# Load taxi trips from the database
	columns = {
		'locs': ["_".join(c) for c in product(["pickup", "dropoff"], ["latitude", "longitude"])],
		'time': ["pickup_datetime", "dropoff_datetime"],
		'dist': ["trip_distance/m"],
	}

	# Sanity/safety queryset limit
	query_limit = 111111

	# Comma-separated list of all [column]s
	columns_as_sql = ", ".join(("[" + c + "]") for c in chain.from_iterable(columns.values()))

	sql = F"""
		SELECT {columns_as_sql} 
		FROM [{table_name}] 
		{where}
		LIMIT {query_limit}
	"""

	logger.debug("Reading the database")
	with sqlite3.connect(PARAM['taxidata']) as con:
		trips = pd.read_sql_query(
			sql=sql,
			con=con,
			parse_dates=["pickup_datetime", "dropoff_datetime"],
		)

	if (len(trips) >= query_limit):
		logger.warning(F"SQL query limit hit ({len(trips)} records)?")

	return trips


def project(trips: pd.DataFrame, graph: nx.DiGraph):
	# Nearest-node computer
	logger.debug("Computing nearest in-graph nodes")
	nearest_node = NearestNode(graph)
	# Note: U.index ~ graph node id  and  U.values ~ distance to node
	U = nearest_node(list(zip(trips['pickup_latitude'], trips['pickup_longitude'])))
	V = nearest_node(list(zip(trips['dropoff_latitude'], trips['dropoff_longitude'])))

	# In-graph node estimates of pickup and dropoff
	projected = pd.DataFrame(index=trips.index, data={'u': U.index, 'v': V.index})

	# Distance from given lat/lon to nearest in-graph lat/lon
	projected = projected.loc[(U.values <= PARAM['tol_nearest_nodes/m']) & (V.values <= PARAM['tol_nearest_nodes/m'])]

	return projected


# ~~~~ #

def compute_metric(graph: nx.DiGraph, trips: pd.DataFrame, callback=None):
	nodes = pd.DataFrame(data=nx.get_node_attributes(graph, name="loc"), index=["lat", "lon"]).T
	edges = pd.DataFrame(pd.Series(data=nx.get_edge_attributes(graph, name="len"), name="len"))
	trips = trips.join(project(trips, graph), how="inner")

	# Only nontrivial trips that are not too long
	trips = trips[trips['u'] != trips['v']]
	trips = trips[trips['trip_distance/m'] <= 7000]

	logger.debug(F"Using {len(trips)} trips")

	# Effective metric, to be modified
	edges['met'] = edges.len.copy()

	for r in range(25):
		logger.debug(F"Round {r}")

		if sum(np.sum(~edges.notna())):
			logger.warning(F"There are edges of n/a metric")

		logger.debug("Computing paths")
		nx.set_edge_attributes(graph, dict(edges.met), name="met")

		trips[['path', 'dist']] = pd.DataFrame(data=parallel_map(GraphPathDist(graph, weight="met"), zip(trips.u, trips.v)), index=trips.index)
		trips['f'] = trips['trip_distance/m'] / trips['dist']

		logger.debug("Computing weight correction")
		correction = pd.Series(index=edges.index, data=1)
		i = (0.9 < trips.f) & (trips.f < 1.1)
		for (path, f) in zip(trips.path[i], trips.f[i]):
			correction[list(pairwise(path))] *= f

		if (1 <= correction.min() <= correction.max() <= 1):
			logger.warning("Trivial weight correction")

		# Clip and apply correction
		correction = 2 ** (0.1 * np.maximum(-1, np.minimum(+1, np.log2(correction))))
		edges.met = edges.met * correction

		# Shave off the extremes
		edges.met = np.maximum(edges.len / 4, edges.met)
		edges.met = np.minimum(edges.len * 4, edges.met)
		edges.met = edges.met * (2 ** (0.01 * np.log2(edges.len / edges.met)))

		if callback:
			callback(SimpleNamespace(graph=graph, nodes=nodes, edges=edges, trips=trips, round=r))

	return edges


def manhattan_metric(table_name):

	# Load and sanitize the road graph
	graph = get_road_graph()

	# Get the database table timespan
	with sqlite3.connect(PARAM['taxidata']) as con:
		sql = F"SELECT min(pickup_datetime), max(dropoff_datetime) FROM [{table_name}]"
		dates = pd.date_range(*pd.read_sql_query(sql=sql, con=con).iloc[0])

	#
	for ((weekday, days), hour) in product(dates.groupby(dates.weekday).items(), range(0, 24)):
		logger.debug(F"weekday/hour/#days = {weekday}/{hour}/{len(days)}")

		# Output filename
		output_path = makedirs(os.path.join(PARAM['out_manhattan_metric_path'], F"{table_name}/{weekday}/{hour:02}/"))
		output_edges_fn = os.path.join(output_path, "edges_met.pkl")
		output_about_fn = os.path.join(output_path, "edges_met.json")

		if os.path.isfile(output_about_fn):
			# Info file exists, assume that the job is being done
			logger.info(F"File {output_about_fn} exists -- skipping")
			continue
		else:
			# Touch info file to reserve the job
			with open(output_about_fn, 'w'):
				pass

		where = "WHERE ({})".format(
			" OR ".join(
				"(('{}' <= pickup_datetime) AND (dropoff_datetime < '{}'))".format(
					day + pd.Timedelta(hour + 0, unit='h'),
					day + pd.Timedelta(hour + 1, unit='h')
				)
				for day in days
			)
		)

		trips = get_trips(table_name, graph, where=where)

		def manhattan_metric_callback(info: SimpleNamespace):
			with open(output_edges_fn, 'wb') as fd:
				pickle.dump(info.edges['met'], fd)
			with open(output_about_fn, 'w') as fd:
				json.dump({'days': list(map(str, days)), 'weekday': weekday, 'hour': hour, '#trips': len(info.trips), 'round': info.round}, fd)
			with nx_draw_met_by_len(info.graph, info.nodes, info.edges) as (fig, ax1):
				fig.savefig(makedirs(os.path.join(output_path, F"UV/{info.round:04}.jpg")), **{**PARAM['savefig_args'], 'dpi': 180})

		compute_metric(graph, trips, manhattan_metric_callback)


# ~~~~ ENTRY ~~~~ #

def main():
	manhattan_metric("yellow_tripdata_2016-05")
	return

	tables = {"green_tripdata_2016-05", "yellow_tripdata_2016-05"}

	for table_name in sorted(tables):
		logger.info("Table {table_name}")

		iterate(table_name)
		# trip_trajectories_ingraph(table_name)


if __name__ == '__main__':
	main()

