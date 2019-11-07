# RA, 2019-11-05

from helpers.commons import parallel_map
from helpers.graphs import largest_component, ApproxGeodistance

from b_models import a_manhattan_metric

import pickle

import numpy as np
import pandas as pd
import networkx as nx

from itertools import product
from more_itertools import pairwise

from progressbar import progressbar

from mpl_toolkits import mplot3d
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import seaborn as sb


PARAM = {
	'road_graph': "../a_prepare_data/data/road_graph/UV/nx_digraph_naive.pkl",
	# 'taxidata': "../a_prepare_data/data/taxidata/sqlite/UV/db.db",

	'edges_met': "../b_models/manhattan_metric/yellow_tripdata_2016-05/1/08/edges_met.pkl",
}

graph = largest_component(pickle.load(open(PARAM['road_graph'], 'rb')))
nx.set_edge_attributes(graph, name="met", values=dict(pd.read_pickle(PARAM['edges_met'])))

sql = dict(
	table_name="yellow_tripdata_2016-05",
	where="('2016-05-02 08:00' <= pickup_datetime) and (pickup_datetime <= '2016-05-02 08:20')",
	limit=100,
)

trips = a_manhattan_metric.get_taxidata_trips(**sql)
trips = trips.join(a_manhattan_metric.project(trips, graph), how="inner")

# Attach estimated trajectories of trips
with a_manhattan_metric.GraphPathDist(graph, edge_weight="met") as gpd:
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
		self.approx_dist = ApproxGeodistance(graph, location_attr="loc")

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

		return coo

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		return False

	def proximity0(self, traj3a: pd.DataFrame, traj3b: pd.DataFrame):
		# assert (traj3a.shape == (len(traj3a), 3))
		# assert (traj3b.shape == (len(traj3b), 3))

		# pd.merge(df1, df2, how="outer", suffixes=["_1", "_2"], left_index=True, right_index=True)
		# pd.concat([df1, df2], axis=1).sort_index().interpolate(limit_area='inside')

		# Note: Cannot do this because `interpolate` fails on object-dtype
		# traj3a = pd.Series(index=traj3a.index, data=list(traj3a.to_numpy()), name="loc")
		# traj3b = pd.Series(index=traj3b.index, data=list(traj3b.to_numpy()), name="loc")


		traj3: pd.DataFrame
		traj3 = pd.merge(traj3a, traj3b, how="outer", suffixes=["_a", "_b"], left_index=True, right_index=True)
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

	M = np.zeros((len(traj3), len(traj3)))
	for ((i, ta), (j, tb)) in progressbar(list(product(enumerate(traj3), repeat=2))):
		M[i, j] = spacetime_traj.proximity0(ta, tb)

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

