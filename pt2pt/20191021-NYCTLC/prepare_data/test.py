
# import multiprocessing
from shapely import geometry
#
# pool = multiprocessing.Pool()
#
# x = pool.map(geometry.Point, zip(range(10), range(10)))
# print(x)

import json

from zipfile import ZipFile
with ZipFile("data/osm/manhattan/osm_json.zip", mode='r') as archive:
	with archive.open("data", mode='r') as fd:
		j = json.load(fd)

j = j['elements']

from collections import Counter
print("Element:", Counter(x['type'] for x in j))

nodes = [x for x in j if (x['type'] == "node")]
ways = [x for x in j if (x['type'] == "way")]

import pandas as pd
nodes = pd.DataFrame(data=nodes).drop(columns=['type']).set_index('id', verify_integrity=True)
ways = pd.DataFrame(data=ways).drop(columns=['type']).set_index('id', verify_integrity=True)

df = pd.DataFrame(columns=["highway", "walkable", "drivable", "cyclable"])
df['highway'] = Counter([tags.get('highway') for tags in ways['tags']]).keys()
df = df.set_index("highway")
df.to_csv("data/osm/highways/highways.csv")
exit(9)


print("Highway:", Counter(tags.get('highway') for tags in ways['tags']))
ways = ways[[(type(tags) is dict) for tags in ways['tags']]]
ways = ways[[bool(tags.get('highway')) for tags in ways['tags']]]

import matplotlib.pyplot as plt
(fig, ax1) = plt.subplots()
for way in ways['nodes'].sample(3000):
	(y, x) = nodes.loc[way, ['lat', 'lon']].values.T
	ax1.plot(x, y, 'k-', lw=0.3)
plt.show()


# from shapely.geometry import shape, GeometryCollection
# shape(j[0])
