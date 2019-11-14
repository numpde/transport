# RA, 2019-11-07

from helpers.commons import Section, parallel_map, Axes
from helpers.graphs import largest_component, ApproxGeodistance

import a_effective_metric_manhattan as manhattan

import os
import json
import pickle

import numpy as np
import pandas as pd
import networkx as nx

from itertools import product
from more_itertools import first

from progressbar import progressbar

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

PARAM = {
	'road_graph': "../data_preparation/data/road_graph/UV/nx_digraph_naive.pkl",
	# 'taxidata': "../data_preparation/data/taxidata/sqlite/UV/db.db",

	# 'edges_met': "../models/manhattan_metric/yellow_tripdata_2016-05/1/08/edges_met.pkl",
}


def get_model():
	model = Pipeline(
		steps=[
			("scaler", StandardScaler()),
			(
				"neural",
				MLPRegressor(
					max_iter=20000,
					hidden_layer_sizes=[8, 8, 8, 8],
					# solver="sgd", learning_rate="adaptive", learning_rate_init=1e-5,
					verbose=True,
				)
			),
		],
		memory=None,
		verbose=False,
	)
	return model


def get_graph() -> nx.DiGraph:
	graph = largest_component(pickle.load(open(PARAM['road_graph'], 'rb')))
	# nx.set_edge_attributes(graph, name="met", values=dict(pd.read_pickle(PARAM['edges_met'])))
	return graph


def get_simulated_trips(graph: nx.DiGraph, ntrips=10000, edge_weight="len"):
	random_state = np.random.RandomState(1)

	node_ids = pd.Series(graph.nodes)
	trips = pd.DataFrame(
		data=list(tuple(node_ids.sample(2, random_state=random_state)) for __ in range(ntrips)),
		columns=["u", "v"],
	)

	with manhattan.GraphPathDist(graph, edge_weight=edge_weight) as pathdist:
		trips = trips.join(
			pd.DataFrame(
				data=parallel_map(pathdist, progressbar(list(zip(trips.u, trips.v)))),
				index=trips.index,
				columns=["path", "distance"],
			),
			how="inner",
		)

	# trips = trips.drop(columns=['path'])

	nodes_loc = nx.get_node_attributes(graph, name="loc")

	for (prefix, nn) in [("pickup", trips.u), ("dropoff", trips.v)]:
		trips = trips.join(
			pd.DataFrame(
				data=list(nodes_loc[n] for n in nn),
				index=trips.index,
				columns=[(prefix + "_" + postfix) for postfix in ["latitude", "longitude"]],
			),
			how="inner",
		)

	return trips


graph = get_graph()
trips = get_simulated_trips(graph)

# with section("Get trips"):
# 	sql = dict(
# 		table_name="yellow_tripdata_2016-05",
# 		where="('2016-05-02 08:00' <= pickup_datetime) and (pickup_datetime <= '2016-05-02 09:00')",
# 		limit=20000,
# 	)
#
# 	trips = manhattan.get_taxidata_trips(**sql)
#
# 	trips['trip_duration'] = trips['dropoff_datetime'] - trips['pickup_datetime']


trips_columns = {
	'locs': ["_".join(c) for c in product(["pickup", "dropoff"], ["latitude", "longitude"])],
	'time': ["pickup_datetime", "dropoff_datetime"],
	'dist': ["distance"],
}

# Keep only shorter trips
trips = trips[trips['distance'] <= 5e3]
trips = trips[trips['distance'] >= 100]
# trips = trips[trips['trip_duration'] >= pd.Timedelta(minutes=1)]
# trips = trips[trips['trip_duration'] <= pd.Timedelta(hours=2)]

print(trips)

# Convert to numeric data type
# trips = trips.apply(pd.to_numeric, axis=0)

features = trips_columns['locs']
# outcome = 'trip_duration'
outcome = 'distance'

df0: pd.DataFrame
df1: pd.DataFrame
(df0, df1) = train_test_split(trips, train_size=0.8, random_state=2)

model = get_model()
model = model.fit(df0[features], df0[outcome])

METER_PER_MILE = 1609.344


# Predicted vs true distance
(fig, ax0) = plt.subplots()
ax0.plot(df1[outcome], model.predict(df1[features]), '.', alpha=0.3)
ax0.set_xlabel(F"True {outcome}")
ax0.set_ylabel(F"Predicted {outcome}")
ax0.grid()
plt.show()


# Plot the trajectories with highest relative inaccuracy of prediction
#
# df1 = df1.assign(predicted=model.predict(df1[features]))
# df1 = df1.assign(mismatch=np.abs(np.log2(df1[outcome] / df1['predicted'])))
#
# style = {'legend.fontsize': 6}
# with mpl.rc_context(style):
# 	with Axes() as ax:
# 		node_loc = nx.get_node_attributes(graph, name="loc")
# 		for (i, trip) in df1.nlargest(n=20, columns=['mismatch']).iterrows():
# 			(y, x) = np.array([node_loc[n] for n in trip['path']]).T
# 			label = F"{round(trip['predicted'])}(?) vs {round(trip[outcome])}(!)"
# 			ax.plot(x, y, label=label)
# 			ax.legend()
#
# 		plt.show()
