
import networkx as nx


IFILE = {
	'OSM' : "OUTPUT/00/kaohsiung.osm",
}

# Requires the "pyosmium" package
# Pyosmium is available under the BSD 2-Clause License
# https://github.com/osmcode/pyosmium
import osmium

# Used some hints from
# http://www.patrickklose.com/posts/parsing-osm-data-with-python/

class RoadNetworkExtractor(osmium.SimpleHandler) :
	def __init__(self) :
		osmium.SimpleHandler.__init__(self)

	def node(self, n) :
		
		# n.location is "a geographic coordinate in WGS84 projection"
		# https://docs.osmcode.org/pyosmium/latest/ref_osm.html#osmium.osm.Location

		self.locs[n.id] = n.location

	def way(self, w) :
		filter_tag = 'highway'

		if (filter_tag in w.tags) :
			self.way_nt[w.id] = { 
				'nodes' : [n.ref for n in w.nodes], 
				'tags' : { t.k : t.v for t in w.tags }
			}

	def relation(self, r) :
		pass

	def apply_file(self, *args, **kwargs) :
		# TODO: Throw Exception
		import sys
		sys.exit("Error: Use the wrapper member 'get_graph' instead")
		pass
	
	def get_graph(self, filename) :
		self.locs = {}
		self.way_nt = {}
		osmium.SimpleHandler.apply_file(self, filename)
		
		G = nx.DiGraph()

		# Step 1: construct edges of the graph

		for (way_id, nt) in self.way_nt.items() :
			(way_nodes, way_tags) = (nt['nodes'], nt['tags'])

			# look for an affirmative 'oneway' tag
			if (way_tags.get('oneway', 'no') == 'yes') :
				G.add_path(way_nodes)
			else : 
				G.add_path(way_nodes)
				G.add_path(reversed(way_nodes))

		# Step 2: assign coordinates to vertices of the graph

		for i in G.nodes() :
			G.node[i]['coords'] = (self.locs[i].lon, self.locs[i].lat)

		return G


p = RoadNetworkExtractor()
G = p.get_graph(IFILE['OSM'])

nodes = list(G.nodes())
pos = { i : G.node[i]['coords'] for i in G.nodes() }
nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes, node_size=1)

import matplotlib.pyplot as plt
plt.show()

