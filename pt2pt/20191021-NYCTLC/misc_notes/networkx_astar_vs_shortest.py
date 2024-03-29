import numpy as np
import pandas as pd
import networkx as nx

from helpers.graphs import odd_king_graph
from helpers.commons import Section

g = odd_king_graph(128, 128)
nodes = pd.DataFrame(data=nx.get_node_attributes(g, name="pos"), index=["lon", "lat"]).T

N = 1000

node_loc = nodes.to_numpy()


def d(u, v):
	(ux, uy) = node_loc[u]
	(vx, vy) = node_loc[v]
	return 3 * ((ux - vx) ** 2 + (uy - vy) ** 2) ** (1 / 2)


with Section("astar_path", out=print):
	for n in range(N):
		(u, v) = nodes.sample(2).index
		nx.astar_path(g, source=u, target=v, heuristic=d, weight="len")

with Section("shortest_path (dijkstra)", out=print):
	for n in range(N):
		(u, v) = nodes.sample(2).index
		nx.shortest_path(g, source=u, target=v, weight="len", method="dijkstra")

with Section("shortest_path (bellman-ford)", out=print):
	for n in range(N):
		(u, v) = nodes.sample(2).index
		nx.shortest_path(g, source=u, target=v, weight="len", method="bellman-ford")

with Section("astar_path (null heuristics)", out=print):
	for n in range(N):
		(u, v) = nodes.sample(2).index
		nx.astar_path(g, source=u, target=v, weight="len")
