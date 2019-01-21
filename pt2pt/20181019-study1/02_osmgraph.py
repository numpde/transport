
# RA, 2018-10-20

## ================== IMPORTS :

from helpers import commons
from helpers import graph

import osmium
import pickle
import networkx as nx


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
	'regions' : [
		"TPE101",
		"kaohsiung",
	],

	# Partition long edges
	'max_edge_len' : 50, # meters

	# Construct the road graph from these tags only
	# https://wiki.openstreetmap.org/wiki/Key:highway
	'osm_way_keep_if_has' : {
		'highway' : ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "service"] + ["motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"],
	},

	'osm_rel_keep_if_has' : {
		'type' : ["route", "route_master"],
	},
}


## ====================== AUX :

open = commons.logged_open


## =================== SLAVES :

class RoadNetworkExtractor(osmium.SimpleHandler) :
	def __init__(self) :
		osmium.SimpleHandler.__init__(self)

	def init_containers(self) :
		# For OSM nodes
		self.node_tags = {}
		self.node_locs = {}
		# For OSM ways
		self.way_tags = {}
		self.way_nodes = {}
		# For OSM relations
		self.rels = {}

	# Callback: new OSM node
	def node(self, n) :
		self.node_tags[n.id] = { t.k : t.v for t in n.tags }
		self.node_locs[n.id] = (n.location.lat, n.location.lon)

	# Callback: new OSM way
	def way(self, w) :
		if not any(w.tags.get(key) in values for (key, values) in PARAM['osm_way_keep_if_has'].items()) :
			return

		# Way tags
		self.way_tags[w.id] = { t.k : t.v for t in w.tags }

		# Forward way
		forward = tuple(n.ref for n in w.nodes)

		if ("yes" == self.way_tags[w.id].get('oneway')) :
			# An affirmative 'oneway' tag found
			self.way_nodes[w.id] = { forward }
		else :
			# Record forward and backward way
			self.way_nodes[w.id] = { forward, tuple(reversed(forward)) }

	# Callback: new OSM relation
	def relation(self, r) :
		if not any(r.tags.get(key) in values for (key, values) in PARAM['osm_rel_keep_if_has'].items()) :
			return

		# Relation tags in rel['t']
		rel = { 't' : { t.k : t.v for t in r.tags } }

		# Relation members by OSM type (relations, nodes, ways)
		for t in ['r', 'n', 'w'] :
			rel[t] = [ m.ref for m in r.members if (m.type == t) ]

		self.rels[r.id] = rel

	# Overloading osmium.SimpleHandler.apply_file
	def apply_file(self, *args, **kwargs) :
		raise RuntimeError("Use the wrapper member 'process' instead")

	# Process an OSM file
	def process(self, filename) :

		# Step 0: read map file into buffers

		self.init_containers()

		commons.logger.info("Reading OSM file...")

		osmium.SimpleHandler.apply_file(self, filename)

		commons.logger.info("Done. Now making the graph...")

		# Step 1: insert all nodes as vertices of the graph

		self.G = nx.DiGraph()

		# We do not add *all* the OSM nodes
		# The nodes that support roads will be added automatically
		# self.G.add_nodes_from(self.node_locs.keys())

		# Step 2: construct edges of the road graph

		# Iterate over way IDs
		for wid in self.way_nodes :

			# Attach those attributes to all segments of the way
			pathattr = { 'wid' : wid, **self.way_tags[wid] }

			# If the key 'wid' was among the tags...
			if not (pathattr['wid'] == wid) :
				commons.logger.warning("Way ID '{}' was not properly recorded in the graph".format(wid))

			# Make graph edges from way
			for nodes in self.way_nodes[wid] :
				self.G.add_path(nodes, **pathattr)

		# Set geo-coordinates of the graph nodes
		# No new nodes are introduced to the graph
		nx.set_node_attributes(self.G, self.node_locs, 'pos')

		# Step 3: compute lengths of graph edges (in meters)

		# Compute edge length as approximate physical distance, in meters
		# Location expected as a (lat, lon) pair
		nx.set_edge_attributes(
			self.G,
			{
				(a, b) : commons.geodesic(self.node_locs[a], self.node_locs[b])
				for (a, b) in self.G.edges()
			},
			name='len'
		)

		return {
			# Road network as a networkx graph
			'G'         : self.G,

			# OSM node ID --> Node tags dictionary
			'node_tags' : self.node_tags,

			# OSM node ID --> (lon, lat)
			'node_locs' : self.node_locs,

			# OSM way ID --> Way tags dictionary
			'way_tags'  : self.way_tags,

			# OSM way ID --> Set containing the OSM way as a tuple, and possibly its reverse
			'way_nodes' : self.way_nodes,

			# OSM relation ID --> Info type (one of 'n'odes, 'w'ays, 'r'elations and 't'ags) --> List or Dict
			'rels'      : self.rels,
		}


# # Example of using the above class
# def illustration() :
#
# 	OSM = RoadNetworkExtractor().get_graph(IFILE['OSM'].format(region="kaohsiung"))
#
# 	# Draw a bus route by its name
# 	# route_name = "建國幹線(返程)" # the route should have the number 88
# 	route_name = "0南路" # circular route
#
# 	import matplotlib.pyplot as plt
#
# 	plt.ion()
# 	plt.show()
#
# 	def pathify(wnodes) :
# 		return ([] and list(zip(list(wnodes)[:-1], list(wnodes)[1:])))
#
# 	for r in OSMrels['route'].values() :
#
# 		if not (r['t'].get('route') == "bus") : continue
#
# 		if not (r['t'].get('name') == route_name) : continue
#
# 		try :
# 			if len(r['n']) :
# 				nx.draw_networkx_nodes(G, pos=node_locs, nodelist=r['n'], node_size=10)
# 				nx.draw_networkx_nodes(G, pos=node_locs, nodelist=r['n'][0:1], node_size=40)
#
# 			for i in r['w'] :
# 				e = pathify(way_nodes[i][True]) + pathify(way_nodes[i][False])
# 				nx.draw_networkx_edges(G, pos=node_locs, edgelist=e, arrows=False)
#
# 		except nx.NetworkXError as e :
# 			# Happens if nonexisting nodes or ways
# 			# are referenced by the relation
# 			commons.logger.warning("networkx error: {}".format(e))
#
# 	input("Press ENTER")


# Extract roads and routes, write to file
def extract(region) :
	commons.logger.info("I. Processing the OSM file...")

	# Parse the OSM file
	OSM = RoadNetworkExtractor().process(IFILE['OSM'].format(region=region))

	commons.logger.info("II. Partitioning long edges...")

	# Break long edges into shorter ones
	graph.break_long_edges(OSM['G'], max_len=PARAM['max_edge_len'])

	# First pickle to disk without KNN tree, then with
	for pickle_with_knn in [False, True] :

		if pickle_with_knn :
			commons.logger.info("III. Making the nearest-neighbor tree for the main component...")

			commons.logger.info(" - Extracting the main component")

			# Restrict to the largest weakly/strongly connected component
			# Note: do not remove edges *after* extracting the connected component
			# Note: non-walkable and non-busable roads are still present
			g = graph.largest_component(OSM['G'])

			# Note: Graph.copy() does not deep-copy container attributes
			# https://networkx.github.io/documentation/latest/reference/classes/generated/networkx.Graph.copy.html

			commons.logger.info(" - Computing the KNN")

			# Main graph component with nearest-neighbor tree
			OSM['main_component_with_knn'] = {
				'g' : g,
				'knn' : graph.compute_geo_knn(nx.get_node_attributes(g, 'pos')),
			}

		# Pickle all to disk
		with open(OFILE['OSM-pickled'].format(region=region), 'wb') as fd :
			pickle.dump(
				{
					**OSM,
					'script' : commons.this_module_body(),
				},
				fd,
				pickle.HIGHEST_PROTOCOL
			)


def show_stats(region) :
	commons.logger.info("Loading the complete graph...")
	G: nx.DiGraph
	G = pickle.load(open(OFILE['OSM-pickled'].format(region=region), 'rb'))['G']

	commons.logger.info("Stats:")

	# Edge lengths
	lens = list(nx.get_edge_attributes(G, 'len').values())
	commons.logger.info("Number of nodes = {}, number of edges = {}".format(G.number_of_nodes(), G.number_of_edges()))
	commons.logger.info("Min edge len = {}, max edge len = {}".format(min(lens), max(lens)))
	# How many nodes need to have short edges
	from math import floor
	H = G.to_undirected()
	for desired_max_len in [5, 10, 20, 50] :
		commons.logger.info("Estimated number of extra nodes to have edges below {}m is {}".format(desired_max_len, sum(floor(s / desired_max_len) for s in nx.get_edge_attributes(H, 'len').values())))
	# Histogram of edge lengths
	import matplotlib.pyplot as plt
	plt.hist(lens, bins=30)
	plt.yscale('log', nonposy='clip')
	plt.xlabel("Edge length")
	plt.ylabel("Number of unidirectional edges")
	plt.show()


## ================== MASTERS :

def extract_all_regions() :
	for region in PARAM['regions'] :
		extract(region)

def show_stats_all_regions() :
	for region in PARAM['regions'] :
		show_stats(region)


## ================== OPTIONS :

OPTIONS = {
	'EXTRACT_ALL' : extract_all_regions,
	'SHOW_STATS' : show_stats_all_regions,
}


## ==================== ENTRY :

if (__name__ == "__main__") :
	commons.parse_options(OPTIONS)
