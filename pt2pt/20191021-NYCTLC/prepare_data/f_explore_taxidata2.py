# RA, 2019-10-27

import os
import math
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import sqlite3
from multiprocessing.pool import Pool
from sklearn.neighbors import BallTree

from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("agg")

makedirs = (lambda fn: (os.makedirs(os.path.dirname(fn), exist_ok=True) or fn))

PARAM = {
	'taxidata': "data/taxidata/sqlite/UV/db.db",
	'road_graph': "data/road_graph/UV/nx_digraph_naive.pkl",

	'out_reported_vs_shortest': makedirs("data/taxidata/exploration/trip_distance_vs_shortest/{table_name}.png"),
	'savefig_args': dict(bbox_inches='tight', pad_inches=0, dpi=300),
}


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


def get_road_graph() -> nx.DiGraph:
	# Load road network graph
	g = pickle.load(open(PARAM['road_graph'], 'rb'))
	# Make sure all shortest paths exist
	# Note: copy() is useful for pickling
	g = g.subgraph(max(nx.strongly_connected_components(g), key=len)).copy()
	return g


def get_trip_data(table_name, graph) -> pd.DataFrame:
	# Load taxi trips from the database
	trips = pd.read_sql_query(
		# sql=F"SELECT * FROM [{table_name}] ORDER BY RANDOM() LIMIT 1000",  # DEBUG
		sql=F"SELECT * FROM [{table_name}] ORDER BY RANDOM() LIMIT 100000",
		con=sqlite3.connect(PARAM['taxidata']),
		parse_dates=["pickup_datetime", "dropoff_datetime"],
	)

	# Nearest-node computer
	nearest_node = NearestNode(graph)
	# On-graph distance computer
	graph_distance = GraphDistance(graph)

	U = nearest_node(list(zip(trips['pickup_latitude'], trips['pickup_longitude'])))
	V = nearest_node(list(zip(trips['dropoff_latitude'], trips['dropoff_longitude'])))

	# In-graph nodes of pickup and dropoff
	trips['shortest'] = list(Pool().imap(
		graph_distance,
		zip(U.index, V.index),
		chunksize=100
	))

	MAX_NEAREST = 20  # meters
	trips = trips.loc[np.logical_and(U.values <= MAX_NEAREST, V.values <= MAX_NEAREST)]

	return trips


def plot_distances(table_name):
	trips = get_trip_data(table_name, get_road_graph())

	# On-graph distance vs reported distance [meters]
	df: pd.DataFrame
	df = pd.DataFrame(data=dict(
		reported=(trips['trip_distance']),
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
		ax1.plot(*zip([0] * 2, [df[['reported', 'shortest']].values.max()] * 2), c='k', ls='--', lw=0.5, zorder=100)
		for (h, hdf) in df.groupby(df['h']):
			c = plt.get_cmap("twilight_shifted")([h / 24])
			ax1.scatter(hdf['reported'], hdf['shortest'], c=c, s=3, alpha=0.8, lw=0, zorder=10, label=(F"{len(hdf)} trips at {h}h"))
		ax1.set_xlabel("Reported distance, km")
		ax1.set_ylabel("Naive graph distance, km")
		ax1.set_xticks(range(11))
		ax1.set_yticks(range(11))
		ax1.legend()
		fig.savefig(PARAM['out_reported_vs_shortest'].format(table_name=table_name), **PARAM['savefig_args'])


def main():
	plot_distances("green_tripdata_2016-05")
	plot_distances("yellow_tripdata_2016-05")


if __name__ == '__main__':
	main()
