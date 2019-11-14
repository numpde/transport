# RA, 2019-11-12

from helpers import graphs, commons, maps

import os
import math

from pathlib import Path

import json
import pickle
import sqlite3

import numpy as np
import pandas as pd
import networkx as nx

from scipy import sparse
from scipy.sparse import linalg

from collections import defaultdict
import itertools
from more_itertools import pairwise

from progressbar import progressbar

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from typing import Tuple

# ~~~~ NOTATION ~~~~ #

HEREPATH = Path(__file__).parent
DATAPATH = Path(HEREPATH / "../data_preparation/data/")


# ~~~~ SETTINGS ~~~~ #

PARAM = {
	'taxidata': (DATAPATH / "taxidata/sqlite/UV/db.db"),
	'road_graph': (DATAPATH / "road_graph/UV/nx_digraph_naive.pkl"),

	'edge_time_attr': "dt",

	# Tolerance for accepting a nearest node (distance in meters)
	'tol_nearest_nodes/m': 20,

	'out_images_path': (HEREPATH / "manhattan_time_to_cross/"),

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


def get_graph() -> nx.DiGraph:
	return graphs.largest_component(pickle.load(open(PARAM['road_graph'], 'rb')))


def get_trips(table, columns="*", where="", order="", limit=30000):
	where = (F"WHERE ({where})" if where else "")
	order = (F"ORDER ({order})" if order else "")
	limit = (F"LIMIT ({limit})" if limit else "")
	sql = F"SELECT {columns} FROM [{table}] {where} {order} {limit}"

	with sqlite3.connect(PARAM['taxidata']) as con:
		trips = pd.read_sql_query(sql, con, parse_dates=["ta", "tb"])

	return trips


def trip_endpoints_on_graph(trips: pd.DataFrame, graph: nx.DiGraph, acceptance_dist=PARAM['tol_nearest_nodes/m']):
	with graphs.GraphNearestNode(graph) as nearest_node:
		# Note: a.index ~ graph node id ,  a.values ~ distance to node
		[a, b] = [nearest_node(trips[[where + "_lat", where + "_lon"]].to_numpy()) for where in ["pickup", "dropoff"]]

	# In-graph node estimates of pickup and dropoff
	endpoints = pd.DataFrame(index=trips.index, data={'a': a.index, 'b': b.index})

	# Distance from given lat/lon to nearest in-graph lat/lon
	endpoints = endpoints.loc[(a.values <= acceptance_dist) & (b.values <= acceptance_dist)]

	return endpoints


# ~~~~

def refine_once(graph: nx.DiGraph, trips: pd.DataFrame):
	edge_dt = pd.Series({
		# Default values
		**{(u, edge_t): (trip_t / 3) for (u, edge_t, trip_t) in graph.edges.data("len")},
		# Overwrite using existing values
		**nx.get_edge_attributes(graph, name=PARAM['edge_time_attr'])
	})

	nx.set_edge_attributes(graph, name=PARAM['edge_time_attr'], values=dict(edge_dt))

	trips = trips.join(trip_endpoints_on_graph(trips, graph), how="inner")

	with graphs.GraphPathDist(graph, edge_weight=PARAM['edge_time_attr']) as pathdist:
		trips = trips.join(pd.DataFrame(
			data=commons.parallel_map(pathdist, progressbar(list(zip(trips.a, trips.b)))),
			columns=["est_path", "est_path_dt"], index=trips.index,
		))

	trips = trips[(trips.duration > 0) & (trips.est_path_dt > 0)]

	trips['f'] = trips.duration / trips.est_path_dt

	def path_edge_matrix(paths, edges) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
		edgej = dict(zip(edges, itertools.count()))
		(ii, jj) = np.array([(ii, edgej[e]) for (ii, path) in enumerate(paths) for e in pairwise(path)]).T
		M = sparse.csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(len(paths), len(edges)))
		return (M, np.unique(ii), np.unique(jj))

	def Vec(a) -> np.ndarray:
		return np.array(a, dtype=float).reshape(-1)

	with commons.Section("Solving LSQR", out=print):
		# Trip x Edge incidence matrix
		(M, nzi, nzj) = path_edge_matrix(trips.est_path, edge_dt.index)

		# *Duration* for each trip
		trip_t = Vec(trips.duration)
		# *Time-to-cross* for each edge, current estimate
		edge_t = Vec(edge_dt)

		#
		S = sparse.diags([d for (u, v, d) in graph.edges.data("len")], format='csc')

		# Not all trips/edges are involved: truncate the linear system
		(M, trip_t) = (M[nzi, :], trip_t[nzi])
		(M, edge_t) = (M[:, nzj], edge_t[nzj])
		S = S[nzj, :][:, nzj]

		assert(len(trip_t) == M.shape[0])
		assert(M.shape[1] == len(edge_t))

		# Compute the correction to the estimate
		# (s, lsqr_istop, lsqr_itn, lsqr_r1norm, lsqr_r2norm, *__) = linalg.lsqr(M, d - M.dot(t), damp=1e-3, show=True)
		(s, *__) = linalg.lsmr(M.dot(S), trip_t - M.dot(edge_t), maxiter=100, damp=1e-1, show=True)
		s = S.dot(s)
		# Update the time-to-cross estimate
		edge_t = np.clip(edge_t + 0.1 * s, a_min=(edge_t / 1.1), a_max=(edge_t * 1.1))

		edge_dt.iloc[nzj] = edge_t

	nx.set_edge_attributes(graph, name=PARAM['edge_time_attr'], values=dict(edge_dt))


# ~~~~ PLOTS ~~~~ #

def speed_on_the_ground(graph: nx.DiGraph, filename: str):
	mpl.use("Agg")

	with commons.Section("Getting the background OSM map"):
		nodes = pd.DataFrame(data=nx.get_node_attributes(graph, name="pos"), index=["lon", "lat"]).T
		extent = maps.ax4(nodes.lat, nodes.lon)
		osmap = maps.get_map_by_bbox(maps.ax2mb(*extent))

	with mpl.rc_context(PARAM['mpl_style']), commons.Axes() as ax1:
		velocity = pd.Series(nx.get_edge_attributes(graph, name="len")) / pd.Series(nx.get_edge_attributes(graph, name=PARAM['edge_time_attr']))
		cmap = LinearSegmentedColormap.from_list(name="noname", colors=["brown", "r", "orange", "g"])

		ax1.imshow(osmap, extent=extent, interpolation='quadric', zorder=-100)

		nx.draw(
			graph,
			ax=ax1,
			pos=nx.get_node_attributes(graph, name="pos"),
			edgelist=list(velocity.index),
			edge_color=list(velocity),
			edge_cmap=cmap,
			edge_vmin=0, edge_vmax=7,
			with_labels=False, arrows=False, node_size=0, alpha=0.8, width=0.3
		)

		fn = PARAM['out_images_path'] / commons.myname() / F"{filename}.png"
		ax1.figure.savefig(commons.makedirs(fn))


# ~~~~ ENTRY ~~~~ #

def main():
	table_name = "yellow_tripdata_2016-05"
	graph = get_graph()
	trips = get_trips(table=table_name, where="('2016-05-02 08:00' <= ta) and (tb <= '2016-05-02 09:00')")

	for i in range(100):
		refine_once(graph, trips)
		speed_on_the_ground(graph, filename=table_name)


if __name__ == '__main__':
	main()
