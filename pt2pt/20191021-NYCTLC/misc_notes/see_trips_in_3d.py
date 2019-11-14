# RA, 2019-11-05

from helpers.commons import parallel_map, Section
from helpers.graphs import largest_component, ApproxGeodistance, GraphPathDist

from models import a_effective_metric_manhattan

import pickle

import numpy as np
import pandas as pd
import networkx as nx

from itertools import product
from more_itertools import pairwise, first, last

from progressbar import progressbar

from mpl_toolkits import mplot3d
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import seaborn as sb


PARAM = {
	'road_graph': "../data_preparation/data/road_graph/UV/nx_digraph_naive.pkl",
	# 'taxidata': "../data_preparation/data/taxidata/sqlite/UV/db.db",

	'edges_met': "../models/manhattan_metric/yellow_tripdata_2016-05/1/08/edges_met.pkl",
}

with Section("Get graph"):
	graph = largest_component(pickle.load(open(PARAM['road_graph'], 'rb')))
	nx.set_edge_attributes(graph, name="met", values=dict(pd.read_pickle(PARAM['edges_met'])))

with Section("Get trips"):
	sql = dict(
		table_name="yellow_tripdata_2016-05",
		where="('2016-05-02 08:00' <= pickup_datetime) and (dropoff_datetime <= '2016-05-02 08:30')",
		limit=200,
	)

	trips = a_effective_metric_manhattan.get_taxidata_trips(**sql)
	trips = trips.join(a_effective_metric_manhattan.project(trips, graph), how="inner")

# Attach estimated trajectories of trips
with a_effective_metric_manhattan.GraphPathDist(graph, edge_weight="met") as gpd:
	trips = trips.join(
		pd.DataFrame(
			data=parallel_map(gpd, progressbar(list(zip(trips.u, trips.v)))),
			index=trips.index,
			columns=["path", "dist"],
		),
		how="inner",
	)


class SpaceTimeTraj:
	def __init__(self, graph: nx.DiGraph):
		self.nodes_loc = nx.get_node_attributes(graph, name="loc")
		self.edges_met = nx.get_edge_attributes(graph, name="met")
		self.pathdist = GraphPathDist(graph, edge_weight="met")
		self.approx_dist = ApproxGeodistance(graph, location_attr="loc")

	def length(self, path):
		return sum(self.edges_met[e] for e in pairwise(path))

	def __call__(self, path, tspan):
		path = tuple(path)
		(t0, t1) = tspan

		tt = np.cumsum([0] + [self.edges_met[e] for e in pairwise(path)])
		tt = (1 / (tt[-1] or 1)) * tt
		tt = [(t0 + t * (t1 - t0)) for t in tt]

		nn = np.array([self.nodes_loc[n] for n in path])

		assert (len(tt) ==len(path))
		assert (nn.shape == (len(path), 2))

		# As a (N x 3) array
		# coo = np.hstack([np.array(tt).reshape(-1, 1), nn])

		coo = pd.DataFrame(index=pd.to_datetime(tt), data=nn, columns=["lat", "lon"])
		coo['node'] = list(path)

		return coo

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		return False

	def proximity1(self, traj1: pd.DataFrame, traj2: pd.DataFrame):
		(a1, b1) = [first(traj1.node), last(traj1.node)]
		(a2, b2) = [first(traj2.node), last(traj2.node)]

		len1 = self.pathdist.dist_only((a1, b1))
		len2 = self.pathdist.dist_only((a2, b2))
		len3 = min(
			sum(self.pathdist.dist_only(uv) for uv in pairwise(inout_sequence))
			for inout_sequence in [
				(a1, a2, b2, b1),
				(a1, a2, b1, b2),
			]
		)
		savings = (len1 + len2) - len3
		savings = savings if (savings > 0) else np.nan
		return savings

	def proximity0(self, traj1: pd.DataFrame, traj2: pd.DataFrame):

		# pd.merge(df1, df2, how="outer", suffixes=["_1", "_2"], left_index=True, right_index=True)
		# pd.concat([df1, df2], axis=1).sort_index().interpolate(limit_area='inside')

		# Note: Cannot do this because `interpolate` fails on object-dtype
		# traj3a = pd.Series(index=traj3a.index, data=list(traj3a.to_numpy()), name="loc")
		# traj3b = pd.Series(index=traj3b.index, data=list(traj3b.to_numpy()), name="loc")

		traj3: pd.DataFrame
		cols = ['lat', 'lon']
		traj3 = pd.merge(traj1[cols], traj2[cols], how="outer", suffixes=["_1", "_2"], left_index=True, right_index=True)
		traj3 = traj3.interpolate('linear')
		traj3 = traj3.dropna()

		assert(traj3.shape[1] == (2 + 2))

		tt = traj3.index.to_pydatetime()

		# Optional -- convert to seconds-offset as float
		tt = [(t - tt[0]).total_seconds() for t in tt]

		ff = [
			self.approx_dist.loc_dist_est(p, q)
			for (p, q) in zip(traj3.iloc[:, 0:2].to_numpy(), traj3.iloc[:, 2:4].to_numpy())
		]

		from scipy.integrate import trapz
		I = trapz(x=tt, y=ff)

		return I

with SpaceTimeTraj(graph) as spacetime_traj:
	traj3 = [
		spacetime_traj(path=trip.path, tspan=(trip.pickup_datetime, trip.dropoff_datetime))
		for (__, trip) in trips.iterrows()
	]

	M = np.nan * np.empty((len(traj3), len(traj3)))
	for ((i, ta), (j, tb)) in progressbar(list(product(enumerate(traj3), repeat=2))):
		if (i != j):
			M[i, j] = spacetime_traj.proximity1(ta, tb)

	sb.heatmap(M)
	plt.show()


# with SpaceTimeTraj(graph) as spacetime_traj:
# 	ax1 = plt.axes(projection='3d')
#
# 	for (i, trip) in trips.iterrows():
# 		traj3 = spacetime_traj(path=trip.path, tspan=(trip.pickup_datetime, trip.dropoff_datetime))
# 		(tt, yy, xx) = traj3.reset_index().to_numpy().T
# 		ax1.plot3D(xx, yy, mdates.date2num(tt), alpha=0.4)
#
# 	plt.show()

