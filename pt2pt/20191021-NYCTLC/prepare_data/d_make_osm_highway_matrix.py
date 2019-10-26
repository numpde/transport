
# RA, 2019-10-26

# https://wiki.openstreetmap.org/wiki/Key:highway


import maps

import pandas as pd
import numpy as np

from itertools import chain

import json
from zipfile import ZipFile

from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("agg")


HW = "highway"

PARAM = {
	'osm_archive': "data/osm/manhattan/osm_json.zip",

	'out_highway_matrix': "data/osm/highways/highway_matrix.csv",
	'out_image': "data/osm/highways/{name}.png",

	'savefig_args': dict(bbox_inches='tight', pad_inches=0, dpi=300),
}


with ZipFile(PARAM['osm_archive'], mode='r') as archive:
	with archive.open("data", mode='r') as fd:
		j = (json.load(fd))['elements']

print("Element:", Counter(x['type'] for x in j))

nodes = pd.DataFrame(data=[x for x in j if (x['type'] == "node")]).set_index('id', verify_integrity=True)
ways = pd.DataFrame(data=[x for x in j if (x['type'] == "way")]).set_index('id', verify_integrity=True)

print("Highway:", Counter(tag.get(HW) for tag in ways['tags']))

# All OSM ways tagged "highway"
ways[HW] = [tags.get(HW) for tags in ways['tags']]
ways = ways[~ways[HW].isna()]
ways = ways.drop(columns=['type'])

# Retain only nodes that support any remaining ways
nodes = nodes.loc[set(chain.from_iterable(ways.nodes.values)), :]

# Collect values of "highway=*"
highway_matrix = pd.DataFrame(index=pd.Index(ways[HW].unique(), name=HW), columns=["drivable", "cyclable", "walkable"])
#
highway_matrix['drivable'][{'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential'}] = True
highway_matrix['drivable'][{'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'}] = True
#
highway_matrix['cyclable'][{'cycleway'}] = True
#
highway_matrix['walkable'][{'footway', 'pedestrian', 'living_street', 'path', 'elevator', 'escalator', 'crossing', 'platform', 'steps'}] = True

highway_matrix.to_csv(PARAM['out_highway_matrix'])


# Make map of highway_matrix

for highway_kind in highway_matrix.columns:
	print(highway_kind)

	# Sub-dataframe of OSM ways
	ways0 = ways[(True == highway_matrix[highway_kind][ways[HW]]).values]

	# print(drivable_ways.groupby(drivable_ways[HW].apply(lambda i: i.split('_')[0])).size().sort_values(ascending=False).index)
	# print(pd.Series(index=(drivable_ways[HW].apply(lambda i: i.split('_')[0]))).groupby(level=0).size().sort_values(ascending=False).index)
	# print(list(reversed(drivable_ways.groupby(HW).size().groupby(lambda i: i.split('_')[0]).sum().sort_values().index)))
	# print(list(drivable_ways.groupby(HW).size().groupby(lambda i: i.split('_')[0]).sum().sort_values(ascending=False).index))
	# exit(9)

	highways = list(ways0.groupby(HW).size().groupby(lambda i: i.split('_')[0]).sum().sort_values(ascending=False).index)

	fig: plt.Figure
	ax1: plt.Axes
	(fig, ax1) = plt.subplots()
	ax1.tick_params(axis='both', which='both', labelsize=3)
	ax1.set_title(F"OSM highways -- {highway_kind}", fontsize=6)

	extent = np.dot(
			[[min(nodes.lon), max(nodes.lon)], [min(nodes.lat), max(nodes.lat)]],
			(lambda s: np.asarray([[1 + s, -s], [-s, 1 + s]]))(0.01)
	).flatten()

	ax1.set_xlim(extent[0:2])
	ax1.set_ylim(extent[2:4])

	for (n, hw) in enumerate(highways):
		label = hw
		for way in ways0[ways0[HW] == hw]['nodes']:
			(y, x) = nodes.loc[way, ['lat', 'lon']].values.T
			ax1.plot(x, y, '-', c=(F"C{n}"), alpha=0.9, lw=0.2, label=label)
			# Avoid thousands of legend entries:
			label = None

	ax1.legend(loc="upper left", fontsize="xx-small")

	# Get the background map
	ax1.imshow(maps.get_map_by_bbox(maps.ax2mb(*extent)), extent=extent, interpolation='quadric', zorder=-100)

	# Save image
	fig.savefig(PARAM['out_image'].format(name=highway_kind), **PARAM['savefig_args'])
	plt.close(fig)

