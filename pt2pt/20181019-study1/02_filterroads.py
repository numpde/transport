
# RA, 2018-10-20

## ================== IMPORTS :

import networkx as nx
import osmium
import pickle
import inspect
import os
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

# Create output directories
for f in OFILE.values() : os.makedirs(os.path.dirname(f), exist_ok=True)


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

		self.locs[n.id] = (n.location.lon, n.location.lat)

	def way(self, w) :
		# Filter out the ways that do not have any of these tags:
		filter_tags = ['highway', 'bridge', 'tunnel']
		if not any(t in w.tags for t in filter_tags) : return

		self.way_nodes[w.id] = [ n.ref for n in w.nodes ]
		self.way_tags [w.id] = { t.k : t.v for t in w.tags }

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
		self.rels[r.tags['type']][r.id] = rel

	def apply_file(self, *args, **kwargs) :
		# TODO: Throw Exception
		import sys
		sys.exit("Error: Use the wrapper member 'get_graph' instead")
		pass
	
	def get_graph(self, filename) :

		# Step 0: read map file into buffers

		self.locs = {}
		self.way_nodes = {}
		self.way_tags = {}
		self.way_edges = defaultdict(dict)
		self.rels = defaultdict(dict)

		osmium.SimpleHandler.apply_file(self, filename)

		# Step 1: insert all nodes as vertices of the graph
		
		self.G = nx.DiGraph()

		self.G.add_nodes_from(self.locs.keys())

		# Step 2: construct edges of the road graph

		def add_path(wnodes, wid, is_forward) :
			wnodes = list(wnodes)
			if not wnodes :
				self.way_edges[wid][is_forward] = [] 
				return
			self.G.add_path(wnodes, wid=wid)
			self.way_edges[wid][is_forward] = list(zip(wnodes[:-1], wnodes[1:]))

		# Iterate over way IDs
		for wid in self.way_nodes.keys() :
			(wnodes, wtags) = (self.way_nodes[wid], self.way_tags[wid])

			# Attach those attributes to all segments of the way
			pathattr = { 'wid' : wid }

			# Note: nx.get_edge_attributes(G, 'wid') returns
			#       a dict of wid's keyed by edge (a, b)

			if (wtags.get('oneway', "?") == "yes") :
				# affirmative 'oneway' tag found
				add_path(wnodes, wid, True)
				add_path([], wid, False)
			else : 
				# add the way forward and backward
				add_path(wnodes, wid, True)
				add_path(reversed(wnodes), wid, False)

		return (self.G, self.locs, self.way_tags, self.way_edges, self.rels)


# Example of using the above class
def illustration() :

	(G, locs, way_tags, way_edges, rels) = RoadNetworkExtractor().get_graph(IFILE['OSM'])

	# Draw a bus route by its name
	# route_name = "建國幹線(返程)" # the route should have the number 88
	route_name = "0南路" # circular route

	import matplotlib.pyplot as plt

	plt.ion()
	plt.show()

	for r in rels['route'].values() :

		if not (r['t'].get('name') == route_name) : continue
		if not (r['t'].get('route') == "bus") : continue

		try :
			if len(r['n']) :
				nx.draw_networkx_nodes(G, pos=locs, nodelist=r['n'], node_size=10)
				nx.draw_networkx_nodes(G, pos=locs, nodelist=r['n'][0:1], node_size=40)

			for i in r['w'] :
				e = way_edges[i][True] + way_edges[i][False]
				nx.draw_networkx_edges(G, pos=locs, edgelist=e, arrows=False)

		except nx.NetworkXError as e :
			# Happens if nonexisting nodes or ways
			# are referenced by the relation
			print(e)

	input()


# Extract roads and routes, write to file
def extract(region) :

	(G, locs, way_tags, way_edges, rels) = (
		RoadNetworkExtractor().get_graph(IFILE['OSM'].format(region=region))
	)

	pickle.dump(
		{
			# Road network as a graph
			'G' : G, 

			# lon-lat location of the graph vertices
			'locs' : locs, 

			# Tags of OSM's ways as a dict, indexed by way ID
			'way_tags' : way_tags, 

			# Edges of the graph for each way, indexed by way ID
			'way_edges' : way_edges, 

			# OSM's relations, index by OSM type, then by ID, 
			# then as 'n'odes, 'w'ays, 'r'elations and 't'ags
			'rels' : rels, 

			# The contents of this script
			'script' : THIS,
		},
		logged_open(OFILE['OSM-pickled'].format(region=region), 'wb'),
		pickle.HIGHEST_PROTOCOL
	)


## ==================== ENTRY :

if (__name__ == "__main__") :
	for region in PARAM['regions'] :
		extract(region)


