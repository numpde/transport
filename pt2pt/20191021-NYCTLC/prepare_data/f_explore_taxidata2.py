# RA, 2019-10-27

import os
import math
import json
import pickle
import inspect

import numpy as np
import pandas as pd
import networkx as nx
import sqlite3

from sklearn.neighbors import BallTree
from multiprocessing.pool import Pool

from collections import Counter
from more_itertools import pairwise

import matplotlib as mpl
import matplotlib.pyplot as plt

import logging as logger
logger.basicConfig(level=logger.DEBUG, format="%(levelname)-8s [%(asctime)s] : %(message)s", datefmt="%Y%m%d %H:%M:%S %Z")
logger.getLogger('matplotlib').setLevel(logger.WARNING)
logger.getLogger('PIL').setLevel(logger.WARNING)

# Local
import maps


# ~~~~ COMMONS ~~~~ #

# Return caller's function name
def myname():
	return inspect.currentframe().f_back.f_code.co_name


# Create path leading to file
def makedirs(filename):
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	return filename


# ~~~~ SETTINGS ~~~~ #

PARAM = {
	'taxidata': "data/taxidata/sqlite/UV/db.db",
	'road_graph': "data/road_graph/UV/nx_digraph_naive.pkl",

	'out_images_path': makedirs("exploration/"),
	'savefig_args': dict(bbox_inches='tight', pad_inches=0, dpi=300),
}


# ~~~~ HELPERS ~~~~ #

class NearestNode:
	RAD_PER_DEGREE = math.pi / 180
	EARTH_RADIUS_METERS = 6367.5 * 1e3

	def __init__(self, graph):
		# Point array
		self.X = pd.DataFrame(data=nx.get_node_attributes(graph, "loc")).T
		# Nearest neighbors tree
		self.T = BallTree(self.X.values * self.RAD_PER_DEGREE, metric="haversine")

	def __call__(self, loc):
		# Get nearest nodes: distance to X and index in X
		(d, i) = np.squeeze(self.T.query(np.asarray(loc) * self.RAD_PER_DEGREE, k=1, return_distance=True))
		# Note: do not sort the Series
		s = pd.Series(index=(self.X.index[list(map(int, i))]), data=(d * self.EARTH_RADIUS_METERS))
		return s


class GraphDistance:
	def __init__(self, graph):
		self.graph = graph

	def __call__(self, uv):
		return nx.shortest_path_length(self.graph, source=uv[0], target=uv[1], weight="len")


class GraphTrajectory:
	def __init__(self, graph):
		self.graph = graph

	def __call__(self, uv):
		return nx.shortest_path(self.graph, source=uv[0], target=uv[1], weight="len")


# ~~~~ DATA SOURCE ~~~~ #

def get_road_graph() -> nx.DiGraph:
	logger.debug(F"{myname()}: Loading the road graph")
	# Load road network graph
	g = pickle.load(open(PARAM['road_graph'], 'rb'))
	# Make sure all shortest paths exist
	# Note: copy() is useful for pickling
	g = g.subgraph(max(nx.strongly_connected_components(g), key=len)).copy()
	return g


def get_trip_data(table_name, graph) -> pd.DataFrame:

	# Load taxi trips from the database
	logger.debug(F"{myname()}: Reading the database")
	trips = pd.read_sql_query(
		# sql=F"SELECT * FROM [{table_name}] ORDER BY RANDOM() LIMIT 1000",  # DEBUG
		# sql=F"SELECT * FROM [{table_name}] ORDER BY RANDOM() LIMIT 10000",  # DEBUG
		sql=F"SELECT * FROM [{table_name}] ORDER BY RANDOM() LIMIT 100000",
		con=sqlite3.connect(PARAM['taxidata']),
		parse_dates=["pickup_datetime", "dropoff_datetime"],
	)

	# Nearest-node computer
	nearest_node = NearestNode(graph)

	logger.debug(F"{myname()}: Computing nearest in-graph nodes")

	# (index, values) correspond to (graph node id, distance)
	U = nearest_node(list(zip(trips['pickup_latitude'], trips['pickup_longitude'])))
	V = nearest_node(list(zip(trips['dropoff_latitude'], trips['dropoff_longitude'])))

	# In-graph node estimates of pickup and dropoff
	trips['u'] = U.index
	trips['v'] = V.index

	# Grace distance from given lat/lon to nearest in-graph lat/lon
	MAX_NEAREST = 20  # meters
	trips = trips.loc[(U.values <= MAX_NEAREST) & (V.values <= MAX_NEAREST)]

	return trips


# ~~~~ PLOTTING ~~~~ #

def trip_distance_vs_shortest(table_name):
	mpl.use("Agg")

	graph = get_road_graph()
	trips = get_trip_data(table_name, graph)

	# On-graph distance between those
	logger.debug(F"{myname()}: Computing shortest distance")
	trips['shortest'] = list(Pool().imap(GraphDistance(graph), zip(trips.u, trips.v), chunksize=100))

	# On-graph distance vs reported distance [meters]
	df: pd.DataFrame
	df = pd.DataFrame(data=dict(
		reported=(trips['trip_distance/m']),
		shortest=(trips['shortest']),
	))
	# Convert to [km] and stay below 10km
	df = df.applymap(lambda x: (x / 1e3))
	df = df.applymap(lambda km: (km if (km < 10) else np.nan)).dropna()

	# Hour of the day
	df['h'] = trips['pickup_datetime'].dt.hour

	style = {'legend.fontsize': 3}  # 'font.size': 10
	with plt.style.context(style):
		fig: plt.Figure
		ax1: plt.Axes
		(fig, ax1) = plt.subplots()
		ax1.set_aspect(aspect="equal", adjustable="box")
		ax1.grid()
		ax1.plot(*(2 * [0, df[['reported', 'shortest']].values.max()]), c='k', ls='--', lw=0.5, zorder=100)
		for (h, hdf) in df.groupby(df['h']):
			c = plt.get_cmap("twilight_shifted")([h / 24])
			ax1.scatter(hdf['reported'], hdf['shortest'], c=c, s=3, alpha=0.8, lw=0, zorder=10, label=(F"{len(hdf)} trips at {h}h"))
		ax1.set_xlabel("Reported distance, km")
		ax1.set_ylabel("Naive graph distance, km")
		ax1.set_xticks(range(11))
		ax1.set_yticks(range(11))
		ax1.legend()

		fn = os.path.join(PARAM['out_images_path'], F"{myname()}/{table_name}.png")
		fig.savefig(makedirs(fn), **PARAM['savefig_args'])


def trip_trajectories_ingraph(table_name):
	mpl.use("Agg")

	# Max number of trajectories to plot
	N = 1000

	graph = get_road_graph()
	nodes = pd.DataFrame(data=nx.get_node_attributes(graph, "loc"), index=["lat", "lon"]).T

	trips = get_trip_data(table_name, graph)

	trips = trips.sample(min(N, len(trips)))
	logger.debug(F"{myname()}: {len(trips)} trips")

	logger.debug(F"{myname()}: Computing trajectories")
	trajectories = Pool().map(GraphTrajectory(graph), zip(trips.u, trips.v))

	# Axes window
	extent = np.dot(
		[[min(nodes.lon), max(nodes.lon)], [min(nodes.lat), max(nodes.lat)]],
		(lambda s: np.asarray([[1 + s, -s], [-s, 1 + s]]))(0.01)
	).flatten()

	logger.debug(F"{myname()}: Getting the background OSM map")
	osmap = maps.get_map_by_bbox(maps.ax2mb(*extent))

	style = {'font.size': 5}
	with plt.style.context(style):
		fig: plt.Figure
		ax1: plt.Axes
		(fig, ax1) = plt.subplots()

		# The background map
		ax1.imshow(osmap, extent=extent, interpolation='quadric', zorder=-100)

		ax1.axis("off")

		ax1.set_xlim(extent[0:2])
		ax1.set_ylim(extent[2:4])

		c = 'b'
		if ("green" in table_name): c = "green"
		if ("yello" in table_name): c = "orange"

		logger.debug(F"{myname()}: Plotting trajectories")
		for traj in trajectories:
			(y, x) = nodes.loc[traj].values.T
			ax1.plot(x, y, c=c, alpha=0.1, lw=0.3)

		# Save to file
		fn = os.path.join(PARAM['out_images_path'], F"{myname()}/{table_name}.png")
		fig.savefig(makedirs(fn), **PARAM['savefig_args'])

		# Meta info
		json.dump({'number_of_trajectories': len(trips)}, open(fn + ".txt", 'w'))


# ~~~~ ENTRY ~~~~ #

def main():
	tables = {"green_tripdata_2016-05", "yellow_tripdata_2016-05"}

	for table_name in sorted(tables):
		logger.info(F"{myname()}: Table {table_name}")

		# trip_distance_vs_shortest(table_name)
		trip_trajectories_ingraph(table_name)


if __name__ == '__main__':
	main()
