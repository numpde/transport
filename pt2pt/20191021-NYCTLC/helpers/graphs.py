# RA, 2019-11-05

from math import ceil

import numpy as np
import networkx as nx

from itertools import groupby
from more_itertools import pairwise


def odd_king_graph(xn=8, yn=8, scale=1.0) -> nx.DiGraph:
	"""
	Constructs a Manhattan-like chessboard directed graph.

	Example:
		import networkx as nx
		import matplotlib.pyplot as plt
		g = king_graph(xn=4, yn=8, scale=2)
		nx.draw(g, pos=nx.get_node_attributes(g, "pos"))
		plt.show()
	"""

	g = nx.DiGraph()

	for i in range(xn):
		for (a, b) in pairwise(range(yn)):
			if (i % 2):
				g.add_edge((i, b), (i, a))
			else:
				g.add_edge((i, a), (i, b))

	for j in range(yn):
		for (a, b) in pairwise(range(xn)):
			if (j % 2):
				g.add_edge((a, j), (b, j))
			else:
				g.add_edge((b, j), (a, j))

	nx.set_node_attributes(g, name="pos", values={n: (n[0] * scale, n[1] * scale) for n in g.nodes})
	nx.set_node_attributes(g, name="loc", values={n: (n[1] * scale, n[0] * scale) for n in g.nodes})
	nx.set_edge_attributes(g, name="len", values=scale)

	g = nx.convert_node_labels_to_integers(g)

	return g


def largest_component(g: nx.DiGraph, components=nx.strongly_connected_components) -> nx.DiGraph:
	"""
	Return the largest connected component as a new graph.
	Uses the copy() method of the graph but does not deep-copy attributes.
	"""

	return nx.subgraph(g, max(components(g), key=len)).copy()


# Split long edges
# Assumes edge length in the edge attribute 'len'
# Assumes node geo-location in the node attribute 'pos'
# Assumes len(u, v) == len(v, u)
def break_long_edges(graph: nx.DiGraph, max_edge_len=50, node_generator=None) -> None:
	len_key = 'len'
	loc_key = 'loc'

	lens = dict(nx.get_edge_attributes(graph, name=len_key))
	assert (lens), F"Expect edge lengths in the '{len_key}' attribute"

	# Edges to split
	edges = {e for (e, s) in lens.items() if (s > max_edge_len)}
	# commons.logger.debug("Number of edges to split is {}".format(len(edges)))

	# Edges are identified and grouped if they connect the same nodes
	iso = (lambda e: (min(e), max(e)))
	edges = [list(g) for (__, g) in groupby(sorted(edges, key=iso), key=iso)]
	# commons.logger.debug("Number of edge groups to split is {}".format(len(edges)))

	class NodeGenerator:
		def __init__(self, g: nx.DiGraph):
			self.g = g
			self.n = 1

		def __iter__(self):
			return self

		def __next__(self):
			while self.g.has_node(self.n):
				self.n += 1
			self.g.add_node(self.n)
			return self.n

	# Get default node ID generator?
	node_generator = node_generator or NodeGenerator(graph)

	def split(e, bothways: bool):
		# Edge to partition
		(a, b) = e
		assert (graph.has_edge(a, b)), "Edge not in graph"

		# Number of new nodes
		nnn = ceil(graph.edges[a, b][len_key] / max_edge_len) - 1

		# Need to partition?
		if not nnn:
			return

		# Relative coordinates of all nodes
		tt = np.linspace(0, 1, 1 + nnn + 1)

		# All new edges have the same length (also the reverse ones)
		new_len = graph.edges[a, b][len_key] / (len(tt) - 1)

		if bothways:
			assert (abs(graph.edges[b, a][len_key] - graph.edges[a, b][len_key]) <= (max_edge_len / 10)), \
				"Gross back and forth edge length mismatch"

		# All nodes along the old edge, old--new--old
		all_nodes = [a] + list(next(node_generator) for __ in range(nnn)) + [b]

		# All new edges
		new_edges = set(zip(all_nodes, all_nodes[1:]))
		if bothways: new_edges |= set(zip(all_nodes[1:], all_nodes))

		# Add the nodes and edges to the graph, copy attributes, overwrite length
		graph.add_edges_from(new_edges, **{**graph.edges[a, b], len_key: new_len})

		# Geo-location of new nodes
		all_pos = map(tuple, zip(*(np.outer(graph.nodes[a][loc_key], 1 - tt) + np.outer(graph.nodes[b][loc_key], tt))))

		# Set the locations of all nodes
		nx.set_node_attributes(graph, values=dict(zip(all_nodes, all_pos)), name=loc_key)

		# if (1 in all_nodes):
		# 	print(nx.get_node_attributes(graph, name=loc_key)[1])

		# Remove old edge(s)
		try:
			graph.remove_edge(a, b)
			graph.remove_edge(b, a)
		except nx.NetworkXError:
			pass

	for ee in edges:
		split(next(iter(ee)), bothways=(len(ee) == 2))
