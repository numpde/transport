# RA, 2019-10-29

from helpers import maps
from helpers.commons import makedirs, Section, parallel_map
from helpers.graphs import GraphPathDist, GraphNearestNode

from inclusive import range

import os
import sys
import math
import json
import pickle
import sqlite3

import numpy as np
import pandas as pd
import networkx as nx

from scipy.sparse import dok_matrix, csc_matrix, csr_matrix

from geopy.distance import great_circle

from dataclasses import dataclass as struct

from time import time as tic
from datetime import datetime, timezone

from types import SimpleNamespace
from typing import Tuple

from itertools import product, chain
from more_itertools import pairwise, first

from contextlib import contextmanager
from retry import retry

from sklearn.neighbors import BallTree as Neighbors

from progressbar import progressbar

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ~~~~ LOGGING ~~~~ #

import logging as logger

logger.basicConfig(
	level=logger.DEBUG,
	format="%(levelname)-8s [%(asctime)s] @%(funcName)s : %(message)s",
	datefmt="%Y%m%d %H:%M:%S %Z",
)
logger.getLogger('matplotlib').setLevel(logger.WARNING)
logger.getLogger('PIL').setLevel(logger.WARNING)


# ~~~~ NOTATION ~~~~ #

def datapath(fn):
	return os.path.join(os.path.dirname(__file__), "../data_preparation/data/", fn)


# ~~~~ SETTINGS ~~~~ #

PARAM = {
	'taxidata': datapath("taxidata/sqlite/UV/db.db"),
	'road_graph': datapath("road_graph/UV/nx_digraph_naive.pkl"),

	# Tolerance for accepting a nearest node (distance in meters)
	'tol_nearest_nodes/m': 20,

	'out_metric': makedirs(os.path.join(os.path.dirname(__file__), "manhattan_metric/")),
	'savefig_args': dict(bbox_inches='tight', pad_inches=0, jpeg_quality=0.9, dpi=300),
}


# ~~~~ HELPERS ~~~~ #




# ~~~~ GRAPHICS ~~~~ #

@contextmanager
@retry(KeyboardInterrupt, tries=2, delay=1)
def nx_draw_met_by_len(graph, edges_met=None, mpl_backend="Agg", printer=None):
	mpl.use(mpl_backend)

	with Section("Preparing to draw graph", out=printer):

		nodes = pd.DataFrame(data=nx.get_node_attributes(graph, name="loc"), index=["lat", "lon"]).T
		edges_len = pd.Series(data=nx.get_edge_attributes(graph, name="len"), name="len")

		if edges_met is not None:
			edges_met = pd.Series(name="met", data=edges_met)
		else:
			edges_met = pd.Series(name="met", data=nx.get_edge_attributes(graph, name="met"))

		cmap = LinearSegmentedColormap.from_list(name="noname", colors=["g", "y", "r", "brown"])

	with Section("Getting the background OSM map", out=printer):
		extent = maps.ax4(nodes.lat, nodes.lon)
		osmap = maps.get_map_by_bbox(maps.ax2mb(*extent))

	with Section("Drawing image", out=printer):

		fig: plt.Figure
		ax1: plt.Axes
		(fig, ax1) = plt.subplots()

		# The background map
		ax1.imshow(osmap, extent=extent, interpolation='quadric', zorder=-100)

		ax1.axis("off")

		edge_colors = (edges_met / edges_len).clip(lower=0.9, upper=1.5)

		nx.draw(
			graph,
			ax=ax1,
			pos=nx.get_node_attributes(graph, name="pos"),
			edge_list=list(edge_colors.index),
			edge_color=list(edge_colors),
			edge_cmap=cmap,
			with_labels=False, arrows=False, node_size=0, alpha=1, width=0.4
		)

		ax1.set_xlim(extent[0:2])
		ax1.set_ylim(extent[2:4])

	try:
		# Note:
		# "yield (fig, ax1)" does not work with the "retry" context manager
		return iter([(fig, ax1)])
	finally:
		plt.close(fig)


# ~~~~ DATA SOURCE ~~~~ #

def get_road_graph() -> nx.DiGraph:
	logger.debug("Loading the initial road graph")
	g = pickle.load(open(PARAM['road_graph'], 'rb'))
	g = g.subgraph(max(nx.strongly_connected_components(g), key=len)).copy()
	return g


def get_taxidata_trips(table_name, where="", orderby="", limit=111111) -> pd.DataFrame:
	# Load taxi trips from the database
	columns = {
		'locs': ["_".join(c) for c in product(["pickup", "dropoff"], ["latitude", "longitude"])],
		'time': ["pickup_datetime", "dropoff_datetime"],
		'dist': ["distance"],
	}

	# Comma-separated list of all [column]s
	columns_as_sql = ", ".join(("[" + c + "]") for c in chain.from_iterable(columns.values()))

	if where:
		where = F"WHERE ({where})"

	if orderby:
		orderby = F"ORDER BY {orderby}"

	sql = F"""
		SELECT {columns_as_sql} 
		FROM [{table_name}] 
		{where}
		{orderby}
		LIMIT {limit}
	"""

	logger.debug("Reading the database")
	with sqlite3.connect(PARAM['taxidata']) as con:
		trips = pd.read_sql_query(
			sql=sql,
			con=con,
			parse_dates=["pickup_datetime", "dropoff_datetime"],
		)

	if (len(trips) >= limit):
		logger.warning(F"SQL query limit hit ({len(trips)} records)?")

	return trips


# ~~~~ TOOLS ~~~~ #

def project(trips: pd.DataFrame, graph: nx.DiGraph):
	# Nearest-node computer
	logger.debug("Computing nearest in-graph nodes")
	nearest_node = GraphNearestNode(graph)
	# Note: U.index ~ graph node id ,  U.values ~ distance to node
	U = nearest_node(list(zip(trips['pickup_latitude'], trips['pickup_longitude'])))
	V = nearest_node(list(zip(trips['dropoff_latitude'], trips['dropoff_longitude'])))

	# In-graph node estimates of pickup and dropoff
	projected = pd.DataFrame(index=trips.index, data={'u': U.index, 'v': V.index})

	# Distance from given lat/lon to nearest in-graph lat/lon
	projected = projected.loc[(U.values <= PARAM['tol_nearest_nodes/m']) & (V.values <= PARAM['tol_nearest_nodes/m'])]

	return projected


@struct
class options_refine_effective_metric:
	num_rounds: int = 70
	temp_graph_metric_attr_name: str = "_met"
	min_trip_distance_m: float = 0.1  # meters
	max_trip_distance_m: float = 1e9  # meters
	correction_factor_moderation: float = 0.8  # learning rate
	random_state: np.random.RandomState = np.random.RandomState(11)


def refine_effective_metric(
		graph: nx.DiGraph,
		trips: pd.DataFrame,
		opt=options_refine_effective_metric(),
		callback=None,
		edges_met=None,
		skip_rounds=0,
) -> pd.Series:
	"""
	Returns a pandas series edges_met such that edges_met[E] is the effective length of edge E.
	If edges_met is provided it is used as the initial guess (but not modified).
	Invalidates the edge attribute opt.temp_graph_metric_attr_name in the graph if present.
	"""

	if nx.get_edge_attributes(graph, name=opt.temp_graph_metric_attr_name):
		logger.warning(F"Graph edge attributes '{opt.temp_graph_metric_attr_name}' will be invalidates")

	# Only nontrivial trips that are not too short or too long
	trips = trips[trips['u'] != trips['v']]
	trips = trips[trips['distance'] >= opt.min_trip_distance_m]
	trips = trips[trips['distance'] <= opt.max_trip_distance_m]

	logger.debug(F"Trip pool has {len(trips)} trips")

	assert ((edges_met is not None) == bool(skip_rounds)), "Both or none of (edges_met, skip_rounds) should be provided"

	# Geographic metric as initial guess / prior
	edges_len = pd.Series(data=nx.get_edge_attributes(graph, name="len"), name="len")

	# Effective metric, to be modified
	if edges_met is not None:
		edges_met = pd.Series(name="met", copy=True, data=edges_met)
		skip_rounds = skip_rounds
	else:
		edges_met = pd.Series(name="met", copy=True, data=nx.get_edge_attributes(graph, name="len"))
		skip_rounds = 0

	for r in range[1 + skip_rounds, opt.num_rounds]:
		logger.debug(F"Round {r}")

		if any(~edges_met.notna()):
			logger.warning(F"There are edges with 'n/a' metric")

		with Section("Computing trajectories", out=logger.debug):

			nx.set_edge_attributes(graph, name=opt.temp_graph_metric_attr_name, values=dict(edges_met))

			with GraphPathDist(graph, edge_weight=opt.temp_graph_metric_attr_name) as gpd:
				# Estimated trajectories of trips
				traj = pd.DataFrame(
					data=parallel_map(gpd, progressbar(list(zip(trips.u, trips.v)))),
					index=trips.index,
					columns=["path", "dist"],
				)

			# Per-trajectory correction factor
			traj['f'] = trips['distance'] / traj['dist']

			# # Accept trips/trajectories that are feasibly related
			# traj = traj[(0.8 < traj.f) & (traj.f < 1.2)]

			logger.debug(F"Weight correction using {sum(traj.f < 1)}(down) + {sum(traj.f > 1)}(up) trips")

		with Section("Computing correction factors", out=logger.debug):

			with Section("Edges of trajectories"):
				edges_loci = dict(zip(edges_met.index, range(len(edges_met))))
				edges_of_traj = list(tuple(edges_loci[e] for e in pairwise(path)) for path in progressbar(traj.path))

			with Section("Incidence matrix [trips x edges]"):
				M = dok_matrix((len(traj), len(edges_met)), dtype=float)
				for (t, edges, f) in zip(range(M.shape[0]), edges_of_traj, traj.f):
					M[t, edges] = f
				del edges_of_traj

			with Section("Subsample trips"):
				I = pd.Series(range(M.shape[0])).sample(frac=0.5, random_state=opt.random_state)
				M = csr_matrix(M)[I, :]

			with Section("Compute correction"):
				M = csc_matrix(M)

				correction = pd.Series(
					index=edges_met.index,
					data=[
						(lambda L: (2 ** np.mean(np.log2(L if len(L) else 1))))(M.getcol(j).data)
						for j in range(M.shape[1])
					]
				).fillna(1)

				# Clip and moderate the correction factors
				correction = 2 ** (opt.correction_factor_moderation * np.log2(correction).clip(lower=-1, upper=+1))

		with Section("Applying correction factors", out=logger.debug):

			edges_met = edges_met * correction

			# Clip extremes, slow-revert to the prior
			edges_met = edges_met.clip(lower=(edges_len / 2), upper=(edges_len * 4))
			edges_met = edges_met * (2 ** (0.01 * np.log2(edges_len / edges_met)))

		if callback:
			# # The edges of estimated trajectories
			# df = pd.DataFrame.sparse.from_spmatrix(
			# 	data=M,
			# 	index=pd.Series(traj.index, name="Estimated trajectory"),
			# 	columns=pd.Index(edges_met.index, name="Edges")
			# ).astype(pd.SparseDtype('float', np.nan))

			callback(SimpleNamespace(graph=graph, trips=trips, edges_met=edges_met, traj=traj, round=r,
									 correction=correction))

	# Record the estimated metric
	nx.set_edge_attributes(graph, name=opt.temp_graph_metric_attr_name, values=dict(edges_met))

	logger.debug(F"Iteration done")

	return edges_met


# ~~~~ WORKER ~~~~ #

def compute_metric_for_table(table_name):
	# Get the database table timespan
	with sqlite3.connect(PARAM['taxidata']) as con:
		sql = F"SELECT min(pickup_datetime), max(dropoff_datetime) FROM [{table_name}]"
		dates = pd.date_range(*pd.read_sql_query(sql=sql, con=con).iloc[0])

	#
	# for ((weekday, days), hour) in product(dates.groupby(dates.weekday).items(), [8]):

	for ((weekday, days), hour) in product(dates.groupby(dates.weekday).items(), range[0, 23]):
		logger.debug(F"weekday/hour = {weekday}/{hour} over {len(days)} days")

		# Filenames for output
		output_path = makedirs(os.path.join(PARAM['out_metric'], F"{table_name}/{weekday}/{hour:02}/"))
		output_edges_fn = os.path.join(output_path, "edges_met.pkl")
		output_about_fn = os.path.join(output_path, "edges_met.json")

		def aboutfile_read():
			try:
				with open(output_about_fn, 'r') as fd:
					return json.load(fd)
			except FileNotFoundError:
				return {'locked': False}

		def aboutfile_write(info):
			with open(output_about_fn, 'w') as fd:
				json.dump(info, fd)

		def manhattan_metric_callback(info: SimpleNamespace):
			assert (info.round is not None)
			assert (info.trips is not None)
			assert (info.edges_met is not None)

			aboutfile_write({**aboutfile_read(), 'valid': False})

			with open(output_edges_fn, 'wb') as fd:
				pickle.dump(info.edges_met, fd)

			aboutfile_write(
				{**aboutfile_read(), 'valid': True, 'days': list(map(str, days)), 'weekday': weekday, 'hour': hour,
				 '#trips': len(info.trips), 'round': info.round,
				 'timestamp': datetime.now(tz=timezone.utc).isoformat()})

			fn = makedirs(os.path.join(output_path, F"UV/{info.round:04}.jpg"))
			if os.path.isfile(fn):
				os.remove(fn)
			with nx_draw_met_by_len(info.graph, info.edges_met) as (fig, ax1):
				fig.savefig(fn, **{**PARAM['savefig_args'], 'dpi': 180})

		# Defaults
		skip_rounds = 0
		edges_met = None

		# Job status
		about = aboutfile_read()

		if about.get('locked', True):
			# Somebody may be working on this now
			logger.info(F"Locked(?) {output_about_fn} exists -- skipping")
			continue

		if (about.get('valid') == False):
			logger.warning(F"Job {output_about_fn} explicitly flagged as invalid -- skipping")
			continue

		if about.get('valid'):
			logger.info(F"Trying to resume {output_about_fn}")
			skip_rounds = about['round']
			with open(output_edges_fn, 'rb') as fd:
				edges_met = pickle.load(fd)

		try:
			# Quickly, reserve the job
			logger.debug("Marking job as locked")
			aboutfile_write({**about, 'locked': True, 'exception': None})

			# Load and sanitize the road graph
			graph = get_road_graph()

			# Taxidata query details
			where = " OR ".join(
				"(('{a}' <= pickup_datetime) AND (dropoff_datetime < '{b}'))".format(
					a=(day + pd.Timedelta(hour + 0, unit='h')),
					b=(day + pd.Timedelta(hour + 1, unit='h')),
				)
				for day in days
			)
			# Query the taxidata database
			trips = get_taxidata_trips(table_name, where=where)

			# Attach nearest nodes
			trips = trips.join(project(trips, graph), how="inner")

			#
			opt = options_refine_effective_metric()
			opt.min_trip_distance_m = 200
			opt.max_trip_distance_m = 7e3
			opt.num_rounds = 100

			# Run the metric computation loop
			# Note: The callback function records the results
			refine_effective_metric(graph, trips, opt=opt, callback=manhattan_metric_callback, skip_rounds=skip_rounds,
									edges_met=edges_met)

		except:
			(exc_type, value, traceback) = sys.exc_info()
			aboutfile_write({**aboutfile_read(), 'exception': (exc_type.__name__)})
			raise
		finally:
			logger.debug("Marking job as non-locked")
			aboutfile_write({**aboutfile_read(), 'locked': False})


# ~~~~ ENTRY ~~~~ #

def main():
	compute_metric_for_table("yellow_tripdata_2016-05")
	return

	tables = {"green_tripdata_2016-05", "yellow_tripdata_2016-05"}

	for table_name in sorted(tables):
		logger.info("Table {table_name}")
		compute_metric_for_table(table_name)


if __name__ == '__main__':
	main()
