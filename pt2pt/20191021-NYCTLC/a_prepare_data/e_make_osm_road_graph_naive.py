
# RA, 2019-10-26

import os

import pandas as pd
import numpy as np
import networkx as nx

from geopy.distance import geodesic as distance

from collections import Counter
from itertools import chain
from more_itertools import pairwise

import pickle

import json
from zipfile import ZipFile

from multiprocessing import Pool

import maps

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("agg")


makedirs = (lambda fn: (os.makedirs(os.path.dirname(fn), exist_ok=True) or fn))


PARAM = {
	'osm_archive': "data/osm/manhattan/osm_json.zip",

	'hw_matrix': pd.read_csv("data/osm/highways/highway_matrix.csv", index_col="highway"),

	'way_tags_we_like': ["name", "highway", "sidewalk", "private", "bus", "cycleway", "oneway", "foot", "pedestrian", "turn"],

	'out_road_graph': makedirs("data/road_graph/UV/nx_digraph_naive.pkl"),

	'out_road_graph_sketch': makedirs("data/road_graph/sketch.png"),
	'savefig_args': dict(bbox_inches='tight', pad_inches=0, dpi=300),
}


with ZipFile(PARAM['osm_archive'], mode='r') as archive:
	J = {
		name: json.load(archive.open("data"))
		for name in archive.namelist()
	}

	assert(1 == len(J))
	J = next(iter(J.values()))
	J = J['elements']

# OSM nodes and OSM ways as DataFrame
nodes: pd.DataFrame
ways: pd.DataFrame
(nodes, ways) = [
	pd.DataFrame(data=(x for x in J if (x['type'] == t))).set_index('id', verify_integrity=True).drop(columns="type")
	for t in ["node", "way"]
]

# Restrict to drivable OSM ways
drivable = PARAM['hw_matrix']['drivable'].fillna(False)
ways = ways.loc[(drivable[tags.get('highway')] for tags in ways['tags']), :]

# Retain only nodes that support any remaining ways
nodes = nodes.loc[set(chain.from_iterable(ways.nodes.values)), :]

# Keep only useful tags
assert("oneway" in PARAM['way_tags_we_like'])
ways.tags = [{k: v for (k ,v) in tags.items() if (k in PARAM['way_tags_we_like'])} for tags in ways.tags]

#
nodes['pos'] = list(zip(nodes['lon'], nodes['lat']))
nodes['loc'] = list(zip(nodes['lat'], nodes['lon']))

#
G = nx.DiGraph()

for (osm_id, way) in ways.iterrows():
	G.add_edges_from(pairwise(way['nodes']), osm_id=osm_id, **way['tags'])
	if not ("yes" == str.lower(way['tags'].get('oneway', "no"))):
		# https://wiki.openstreetmap.org/wiki/Key:oneway
		G.add_edges_from(pairwise(reversed(way['nodes'])), osm_id=osm_id, **way['tags'])

def edge_len(uv):
	return (uv, distance(nodes['loc'][uv[0]], nodes['loc'][uv[1]]).m)

nx.set_edge_attributes(G, name="len", values=dict(Pool().map(edge_len, G.edges)))

nx.set_node_attributes(G, name="pos", values=dict(nodes['pos']))
nx.set_node_attributes(G, name="loc", values=dict(nodes['loc']))


# Save the graph

pickle.dump(G, open(PARAM['out_road_graph'], 'wb'))


# Plot

fig: plt.Figure
ax1: plt.Axes
(fig, ax1) = plt.subplots()
ax1.tick_params(axis='both', which='both', labelsize=3)

extent = np.dot(
	[[min(nodes.lon), max(nodes.lon)], [min(nodes.lat), max(nodes.lat)]],
	(lambda s: np.asarray([[1 + s, -s], [-s, 1 + s]]))(0.01)
).flatten()

nx.draw_networkx(G.to_undirected(), ax=ax1, pos=nx.get_node_attributes(G, "pos"), with_labels=False, arrows=False, node_size=0, alpha=0.9, width=0.3)

ax1.set_xlim(extent[0:2])
ax1.set_ylim(extent[2:4])

# Get the background map
ax1.imshow(maps.get_map_by_bbox(maps.ax2mb(*extent)), extent=extent, interpolation='quadric', zorder=-100)

# Save image
fig.savefig(PARAM['out_road_graph_sketch'], **PARAM['savefig_args'])
plt.close(fig)

