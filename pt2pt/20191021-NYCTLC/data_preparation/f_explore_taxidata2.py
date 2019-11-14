# RA, 2019-10-27

# Local
from helpers import maps
from helpers.commons import myname, makedirs, parallel_map, Section, Axes
from helpers.graphs import largest_component, GraphPathDist, GraphNearestNode

import os
import math
import json
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import sqlite3

from collections import Counter, defaultdict
from more_itertools import pairwise

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import logging as logger

logger.basicConfig(
	level=logger.DEBUG,
	format="%(levelname)-8s [%(asctime)s] : %(message)s",
	datefmt="%Y%m%d %H:%M:%S %Z",
)
logger.getLogger('matplotlib').setLevel(logger.WARNING)
logger.getLogger('PIL').setLevel(logger.WARNING)

# ~~~~ SETTINGS ~~~~ #

PARAM = {
	'taxidata': "data/taxidata/sqlite/UV/db.db",
	'road_graph': "data/road_graph/UV/nx_digraph_naive.pkl",

	'out_images_path': makedirs("exploration/"),

	'mpl_style': {
		'font.size': 3,
		'xtick.major.size': 2,
		'ytick.major.size': 0,
		'xtick.major.pad': 1,
		'ytick.major.pad': 1,

		'savefig.bbox': "tight",
		'savefig.pad_inches': 0,
		'savefig.dpi': 300,
	},
}


# ~~~~ HELPERS ~~~~ #


# ~~~~ DATA SOURCE ~~~~ #

def get_road_graph() -> nx.DiGraph:
	with Section("Loading the road graph", out=logger.debug):
		return largest_component(pickle.load(open(PARAM['road_graph'], 'rb')))


def get_trip_data(table_name, graph, where="", order="random()", limit=100000) -> pd.DataFrame:
	# Load taxi trips from the database
	with Section("Reading the database", out=logger.debug):
		# sql = F"SELECT * FROM [{table_name}] ORDER BY RANDOM() LIMIT 1000"  # DEBUG
		# sql = F"SELECT * FROM [{table_name}] ORDER BY RANDOM() LIMIT 10000"  # DEBUG
		sql = F"SELECT * FROM [{table_name}]"
		sql += (F" WHERE    ({where}) " if where else "")
		sql += (F" ORDER BY ({order}) " if order else "")
		sql += (F" LIMIT    ({limit}) " if limit else "")

		trips = pd.read_sql_query(
			sql=sql,
			con=sqlite3.connect(PARAM['taxidata']),
			parse_dates=["pickup_datetime", "dropoff_datetime"],
		)

		# Trip duration
		trips['duration/s'] = (trips['dropoff_datetime'] - trips['pickup_datetime']).dt.total_seconds()

	with Section("Computing nearest in-graph nodes", out=logger.debug):
		# Nearest-node computer
		nearest_node = GraphNearestNode(graph)

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

	with Section("Computing shortest distances", out=logger.debug):
		trips = trips.join(
			pd.DataFrame(
				data=parallel_map(GraphPathDist(graph, edge_weight="len"), zip(trips.u, trips.v)),
				columns=['path', 'shortest'], index=trips.index,
			)
		)

	# On-graph distance vs reported distance [meters]
	df: pd.DataFrame
	df = pd.DataFrame(data=dict(
		reported=(trips['distance']),
		shortest=(trips['shortest']),
	))
	# Convert to [km] and stay below 10km
	df = df.applymap(lambda x: (x / 1e3))
	df = df.applymap(lambda km: (km if (km < 10) else np.nan)).dropna()

	# Hour of the day
	df['h'] = trips['pickup_datetime'].dt.hour

	with plt.style.context(PARAM['mpl_style']):
		with Axes() as ax1:
			ax1.set_aspect(aspect="equal", adjustable="box")
			ax1.grid()
			ax1.plot(*(2 * [[0, df[['reported', 'shortest']].values.max()]]), c='k', ls='--', lw=0.5, zorder=100)
			for (h, hdf) in df.groupby(df['h']):
				c = plt.get_cmap("twilight_shifted")([h / 24])
				ax1.scatter(
					hdf['reported'], hdf['shortest'],
					c=c, s=3, alpha=0.8, lw=0, zorder=10,
					label=(F"{len(hdf)} trips at {h}h")
				)
			ax1.set_xlabel("Reported distance, km")
			ax1.set_ylabel("Naive graph distance, km")
			ax1.set_xticks(range(11))
			ax1.set_yticks(range(11))
			ax1.legend()

			# Save to file
			fn = os.path.join(PARAM['out_images_path'], F"{myname()}/{table_name}.png")
			ax1.figure.savefig(makedirs(fn))

			# Meta info
			json.dump({'number_of_datapoints': len(df)}, open((fn + ".txt"), 'w'))


def trip_trajectories_ingraph(table_name):
	mpl.use("Agg")

	# Max number of trajectories to plot
	N = 1000

	graph = get_road_graph()
	nodes = pd.DataFrame(data=nx.get_node_attributes(graph, "loc"), index=["lat", "lon"]).T

	trips = get_trip_data(table_name, graph)

	trips = trips.sample(min(N, len(trips)))
	logger.debug(F"{len(trips)} trips")

	logger.debug("Computing trajectories")
	trajectories = parallel_map(GraphPathDist(graph).path_only, zip(trips.u, trips.v))

	with Section("Getting the background OSM map", out=logger.debug):
		extent = maps.ax4(nodes.lat, nodes.lon)
		osmap = maps.get_map_by_bbox(maps.ax2mb(*extent))

	with plt.style.context({**PARAM['mpl_style'], 'font.size': 5}):
		with Axes() as ax1:
			# The background map
			ax1.imshow(osmap, extent=extent, interpolation='quadric', zorder=-100)

			ax1.axis("off")

			ax1.set_xlim(extent[0:2])
			ax1.set_ylim(extent[2:4])

			c = 'b'
			if ("green" in table_name): c = "green"
			if ("yello" in table_name): c = "orange"

			logger.debug("Plotting trajectories")
			for traj in trajectories:
				(y, x) = nodes.loc[list(traj)].values.T
				ax1.plot(x, y, c=c, alpha=0.1, lw=0.3)

			# Save to file
			fn = os.path.join(PARAM['out_images_path'], F"{myname()}/{table_name}.png")
			ax1.figure.savefig(makedirs(fn))

			# Meta info
			json.dump({'number_of_trajectories': len(trips)}, open((fn + ".txt"), 'w'))


def trip_trajectories_velocity(table_name):
	mpl.use("Agg")

	# Max number of trajectories to use
	N = 10000

	graph = get_road_graph()
	nodes = pd.DataFrame(data=nx.get_node_attributes(graph, "loc"), index=["lat", "lon"]).T

	edge_name = pd.Series(nx.get_edge_attributes(graph, name="name"))

	where = "('2016-05-02 08:00' <= pickup_datetime) and (pickup_datetime <= '2016-05-02 09:00')"
	trips = get_trip_data(table_name, graph, order="", limit=N, where=where)

	trips['velocity'] = trips['distance'] / trips['duration/s']
	trips = trips.sort_values(by='velocity', ascending=True)

	logger.debug(F"{len(trips)} trips")

	with Section("Computing estimated trajectories", out=logger.debug):
		trips['traj'] = parallel_map(GraphPathDist(graph).path_only, zip(trips.u, trips.v))

	with Section("Getting the background OSM map", out=logger.debug):
		extent = maps.ax4(nodes.lat, nodes.lon)
		osmap = maps.get_map_by_bbox(maps.ax2mb(*extent))

	with Section("Computing edge velocities", out=logger.debug):
		edge_vel = defaultdict(list)
		for (traj, v) in zip(trips.traj, trips.velocity):
			for e in pairwise(traj):
				edge_vel[e].append(v)
		edge_vel = pd.Series({e: np.mean(v or np.nan) for (e, v) in edge_vel.items()}, index=graph.edges)
		edge_vel = edge_vel.dropna()

	with plt.style.context({**PARAM['mpl_style'], 'font.size': 5}), Axes() as ax1:
		# The background map
		ax1.imshow(osmap, extent=extent, interpolation='quadric', zorder=-100)

		ax1.axis("off")

		ax1.set_xlim(extent[0:2])
		ax1.set_ylim(extent[2:4])

		cmap_velocity = LinearSegmentedColormap.from_list(name="noname", colors=["brown", "r", "orange", "g"])

		# marker = dict(markersize=0.5, markeredgewidth=0.1, markerfacecolor="None")
		# ax1.plot(trips['pickup_longitude'], trips['pickup_latitude'], 'og', **marker)
		# ax1.plot(trips['dropoff_longitude'], trips['dropoff_latitude'], 'xr', **marker)

		# for e in edge_name[edge_name == "65th Street Transverse"].index:
		# 	print(e, edge_vel[e])

		edge_vel: pd.Series
		# edge_vel = edge_vel.rank(pct=True)
		edge_vel = edge_vel.clip(lower=2, upper=6).round()
		edge_vel = (edge_vel - edge_vel.min()) / (edge_vel.max() - edge_vel.min())
		edge_vel = edge_vel.apply(cmap_velocity)

		nx.draw_networkx_edges(
			graph.edge_subgraph(edge_vel.index),
			ax=ax1,
			pos=nx.get_node_attributes(graph, name="pos"),
			edge_list=list(edge_vel.index),
			edge_color=list(edge_vel),
			# edge_cmap=cmap_velocity,
			# vmin=0, vmax=1,
			with_labels=False, arrows=False, node_size=0, alpha=0.8, width=0.3,
		)

		# Save to file
		fn = os.path.join(PARAM['out_images_path'], F"{myname()}/{table_name}.png")
		ax1.figure.savefig(makedirs(fn))

		# Meta info
		json.dump({'number_of_trajectories': len(trips)}, open((fn + ".txt"), 'w'))


def compare_multiple_trajectories(table_name):
	mpl.use("Agg")

	# Number of trips to plot
	N = 10
	# Number of trajectories per trip
	M = 12

	graph = get_road_graph()
	nodes = pd.DataFrame(data=nx.get_node_attributes(graph, "loc"), index=["lat", "lon"]).T
	edges_len = nx.get_edge_attributes(graph, name="len")

	where = "('2016-05-02 08:00' <= pickup_datetime) and (pickup_datetime <= '2016-05-02 09:00')"
	trips = get_trip_data(table_name, graph, order="", where=where)

	trips = trips.sample(min(N, len(trips)), random_state=1)
	logger.debug(F"{len(trips)} trips")

	with Section("Getting the background OSM map", out=logger.debug):
		extent = maps.ax4(nodes.lat, nodes.lon)
		osmap = maps.get_map_by_bbox(maps.ax2mb(*extent))

	with plt.style.context({**PARAM['mpl_style'], 'font.size': 5}), Axes() as ax1:
		# The background map
		ax1.imshow(osmap, extent=extent, interpolation='quadric', zorder=-100)

		ax1.axis("off")

		ax1.set_xlim(extent[0:2])
		ax1.set_ylim(extent[2:4])

		for (__, trip) in trips.iterrows():
			with Section("Computing candidate trajectories", out=logger.debug):
				trajectories = pd.DataFrame(data={'path': [
					path
					for (__, path) in
					zip(range(M), nx.shortest_simple_paths(graph, source=trip.u, target=trip.v))
				]})
				trajectories['dist'] = [sum(edges_len[e] for e in pairwise(path)) for path in trajectories.path]
				trajectories = trajectories.sort_values(by='dist', ascending=False)

			marker = dict(markersize=2, markeredgewidth=0.2, markerfacecolor="None")
			ax1.plot(trip['pickup_longitude'], trip['pickup_latitude'], 'og', **marker)
			ax1.plot(trip['dropoff_longitude'], trip['dropoff_latitude'], 'xr', **marker)

			cmap = LinearSegmentedColormap.from_list(name="noname", colors=["g", "orange", "r", "brown"])
			colors = cmap(pd.Series(trajectories['dist'] / trip['distance']).rank(pct=True))

			for (c, path) in zip(colors, trajectories.path):
				(y, x) = nodes.loc[list(path)].values.T
				ax1.plot(x, y, c=c, alpha=0.5, lw=0.3)

			# Save to file
			fn = os.path.join(PARAM['out_images_path'], F"{myname()}/{table_name}.png")
			ax1.figure.savefig(makedirs(fn))


# ~~~~ ENTRY ~~~~ #

def main():
	tables = {"green_tripdata_2016-05", "yellow_tripdata_2016-05"}

	for table_name in sorted(tables):
		logger.info(F"Table {table_name}")

		# trip_distance_vs_shortest(table_name)
		# trip_trajectories_ingraph(table_name)
		trip_trajectories_velocity(table_name)
		# compare_multiple_trajectories(table_name)


if __name__ == '__main__':
	main()
