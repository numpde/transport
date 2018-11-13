
# RA, 2018-10-20

## ================== IMPORTS :

from helpers import commons
from helpers import graph

import networkx as nx
import osmium
import pickle
import inspect
import time
from collections import defaultdict


## ==================== NOTES :

# Used some hints from
# http://www.patrickklose.com/posts/parsing-osm-data-with-python/

# "Pyosmium" is available under the BSD 2-Clause License
# https://github.com/osmcode/pyosmium

# osmiumnode.location is "a geographic coordinate in WGS84 projection"
# https://docs.osmcode.org/pyosmium/latest/ref_osm.html#osmium.osm.Location

# OSM roads
# https://wiki.openstreetmap.org/wiki/Key:highway


## ==================== INPUT :

IFILE = {
	'OSM' : "OUTPUT/01/UV/{region}.osm",
}


## =================== OUTPUT :

OFILE = {
	'OSM-pickled' : "OUTPUT/02/UV/{region}.pkl",
}

commons.makedirs(OFILE)


## ==================== PARAM :

PARAM = {
	'regions' : ["kaohsiung"],
}


## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))

# Log which files are opened
def logged_open(filename, mode='r', *argv, **kwargs) :
	print("({}):\t{}".format(mode, filename))
	return open(filename, mode, *argv, **kwargs)


## ===================== WORK :

class RoadNetworkExtractor(osmium.SimpleHandler) :
	def __init__(self) :
		osmium.SimpleHandler.__init__(self)

	def node(self, n) :
		self.node_tags[n.id] = { t.k : t.v for t in n.tags }
		self.locs[n.id] = (n.location.lat, n.location.lon)

	def way(self, w) :
		# Filter out the ways that do not have any of these tags:
		#filter_tags = ['highway', 'bridge', 'tunnel']
		# Note: 'tunnel' includes the MRT
		filter_tags = ['highway']
		if not any((t in w.tags) for t in filter_tags) : return

		if 'highway' in w.tags :
			t = w.tags['highway']
			# https://wiki.openstreetmap.org/wiki/Key:highway
			# Exclude "service"
			highway_roads = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential"]
			highway_links = ["motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"]
			if not (t in (highway_roads + highway_links)) : return

		wtags = { t.k : t.v for t in w.tags }
		self.way_tags[w.id] = wtags

		wnodes = [ n.ref for n in w.nodes ]
		# Forward way
		self.way_nodes[w.id][True] = wnodes

		if (wtags.get('oneway', "?") == "yes") :
			# An affirmative 'oneway' tag found
			self.way_nodes[w.id][False] = []
		else :
			# Also add backward way
			self.way_nodes[w.id][False] = list(reversed(wnodes))

	def relation(self, r) :
		# Type of the relation
		r_type = r.tags.get('type')

		# Ignore any relation with unknown type
		if not r_type : return

		# Ignore all but the following types, for now
		if not r_type in ["route", "route_master"] : return

		rel = dict()

		# Relation members, grouped by type
		for t in ['r', 'n', 'w'] :
			rel[t] = [ m.ref for m in r.members if (m.type == t) ]

		# Relation tags
		rel['t'] = { t.k : t.v for t in r.tags }

		# All relations are grouped by type
		self.rels[r_type][r.id] = rel

	def apply_file(self, *args, **kwargs) :
		raise RuntimeError("Use the wrapper member 'get_graph' instead")

	def get_graph(self, filename) :

		# Step 0: read map file into buffers

		self.node_tags = {}
		self.locs = {}
		self.way_nodes = defaultdict(dict)
		self.way_tags = {}
		self.rels = defaultdict(dict)

		print("Reading OSM file...")

		osmium.SimpleHandler.apply_file(self, filename)

		print("Done. Now making the graph...")

		# Step 1: insert all nodes as vertices of the graph

		print(" - 1. Collecting nodes")

		self.G = nx.DiGraph()

		# We do not need all the OSM nodes
		# The nodes that support roads will be added automatically
		# self.G.add_nodes_from(self.locs.keys())

		# Step 2: construct edges of the road graph

		print(" - 2. Collecting edges")

		def add_path(wnodes, wid, is_forward) :
			self.G.add_path(wnodes, wid=wid)
			#self.way_nodes[wid][is_forward] = list(wnodes)

		# Iterate over way IDs
		for wid in self.way_nodes.keys() :
			(wnodes_bothways, wtags) = (self.way_nodes[wid], self.way_tags[wid])

			# Attach those attributes to all segments of the way
			pathattr = { 'wid' : wid }

			# Note: nx.get_edge_attributes(G, 'wid') returns
			#       a dict of wid's keyed by edge (a, b)

			self.G.add_path(wnodes_bothways[True], wid=wid)
			self.G.add_path(wnodes_bothways[False], wid=wid)

			time.sleep(0.001)

		# Step 3: compute lengths of graph edges (in meters)

		print(" - 3. Computing edge lengths")

		# Compute edge length representing distance, in meters
		# Location expected as a (lat, lon) pair
		# https://geopy.readthedocs.io/
		# https://stackoverflow.com/a/43211266
		lens = { (a, b): commons.geodesic(self.locs[a], self.locs[b]) for (a, b) in self.G.edges() }
		nx.set_edge_attributes(self.G, lens, name='len')

		return (self.G, self.node_tags, self.locs, self.way_tags, self.way_nodes, self.rels)


# Example of using the above class
def illustration() :

	(G, node_tags, locs, way_tags, way_nodes, rels) = RoadNetworkExtractor().get_graph(IFILE['OSM'].format(region="kaohsiung"))

	# Draw a bus route by its name
	# route_name = "建國幹線(返程)" # the route should have the number 88
	route_name = "0南路" # circular route

	import matplotlib.pyplot as plt

	plt.ion()
	plt.show()

	def pathify(wnodes) :
		return ([] and list(zip(list(wnodes)[:-1], list(wnodes)[1:])))

	for r in rels['route'].values() :

		if not (r['t'].get('route') == "bus") : continue

		if not (r['t'].get('name') == route_name) : continue

		try :
			if len(r['n']) :
				nx.draw_networkx_nodes(G, pos=locs, nodelist=r['n'], node_size=10)
				nx.draw_networkx_nodes(G, pos=locs, nodelist=r['n'][0:1], node_size=40)

			for i in r['w'] :
				e = pathify(way_nodes[i][True]) + pathify(way_nodes[i][False])
				nx.draw_networkx_edges(G, pos=locs, edgelist=e, arrows=False)

		except nx.NetworkXError as e :
			# Happens if nonexisting nodes or ways
			# are referenced by the relation
			print(e)

	input()

# Extract roads and routes, write to file
def extract(region) :

	print("I. Processing the OSM file")

	(G, node_tags, locs, way_tags, way_nodes, rels) = (
		RoadNetworkExtractor().get_graph(IFILE['OSM'].format(region=region))
	)

	for mode in ['WithoutNN', 'WithNN'] :

		if (mode == 'WithoutNN') :
			main_component_knn = None
		else :
			print("II. Making the nearest-neighbor tree for the main component...")

			# Restrict to the largest weakly/strongly connected component
			g = nx.subgraph(G, max(nx.weakly_connected_components(G), key=len)).copy()

			# Note: Graph.copy() does not deep-copy container attributes
			# https://networkx.github.io/documentation/latest/reference/classes/generated/networkx.Graph.copy.html

			knn = graph.compute_knn(g, locs)

			main_component_knn = {
				'g' : g,
				'node_ids' : knn['node_ids'],
				'knn_tree' : knn['knn_tree'],
			}

		pickle.dump(
			{
				# Road network as a graph
				'G' : G,

				# OSM node info, indexed by node ID
				'node_tags' : node_tags,

				# lon-lat location of the graph vertices
				'locs' : locs,

				# Tags of OSM's ways as a dict, indexed by way ID
				'way_tags' : way_tags,

				# Nodes for each OSM way, indexed by way ID, then by direction True/False
				'way_nodes' : way_nodes,

				# OSM's relations, index by OSM type, then by ID,
				# then as 'n'odes, 'w'ays, 'r'elations and 't'ags
				'rels' : rels,

				# Nearest-neighbor tree
				'main_component_knn' : main_component_knn,

				# The contents of this script
				'script' : THIS,
			},
			logged_open(OFILE['OSM-pickled'].format(region=region), 'wb'),
			pickle.HIGHEST_PROTOCOL
		)


## ==================== ENTRY :

if (__name__ == "__main__") :

	# illustration()

	for region in PARAM['regions'] :
		extract(region)


