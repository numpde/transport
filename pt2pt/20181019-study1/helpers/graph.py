
# RA, 2018-11-11

from helpers import commons
from helpers import maps

import angles
import networkx as nx
import numpy as np
import math
import pickle
import random
import sklearn.neighbors
import geopy.distance
import gpxpy, gpxpy.gpx
from itertools import chain
from sklearn.neighbors import NearestNeighbors

# Metric for (lat, lon) coordinates
def geodist(a, b) :
	return geopy.distance.geodesic(a, b).m

# th is accuracy tolerance in meters
# TH is care-not radius for far-away segments
# Returns a pair (distance, lambda) where
# 0 <= lambda <= 1 is the relative location of the closest point
def dist_to_segment(x, ab, th=5, TH=1000) :
	# Endpoints of the segment
	(a, b) = ab

	# Relative location on the original segment
	(s, t) = (0, 1)

	# Distances to the endpoints
	(da, db) = (geodist(x, a), geodist(x, b))

	while True :

		# Relative accuracy of about 5% achieved?
		if (geodist(a, b) < 0.05 * min(da, db)) :
			break
			#(th < abs(da - db)) and (min(da, db) < TH)

		# Note: potentially problematic at lon~180
		# Approximation of the midpoint (m is not necessarily on a geodesic?)
		m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

		# Distance to the midpoint
		dm = geodist(x, m)

		if (da < db) :
			# Keep bisecting the (a, m) segment
			(b, db, t) = (m, dm, (s + t) / 2)
		else :
			# Keep bisecting the (m, b) segment
			(a, da, s) = (m, dm, (s + t) / 2)

	return min((da, s), (db, t))


def compute_knn(G : nx.DiGraph, locs, leaf_size=150):

	# Only care about the OSM nodes that are in the road network graph
	locs = { i: locs[i] for i in G.nodes() }

	(I, X) = (list(locs.keys()), list(locs.values()))

	return {
		'node_ids': I,
		'knn_tree': sklearn.neighbors.BallTree(X, leaf_size=leaf_size, metric='pyfunc', func=commons.geodesic)
	}

def foo() :

	osm_graph_file = "../OUTPUT/02/UV/kaohsiung.pkl"

	print("Loading OSM...")
	OSM = pickle.load(open(osm_graph_file, 'rb'))

	# Road network (main graph component) with nearest-neighbor tree for the nodes
	knn_tree : NearestNeighbors
	(G, knn_ids, knn_tree) = commons.inspect(('g', 'node_ids', 'knn_tree'))(OSM['main_component_knn'])

	# Locations of the graph nodes
	node_pos = OSM['locs']

	# Free up some memory
	del OSM

	# Get some waypoints
	routes_file = "../OUTPUT/00/ORIGINAL_MOTC/Kaohsiung/CityBusApi_StopOfRoute.json"

	motc_routes = commons.index_dicts_by_key(
		commons.zipjson_load(routes_file),
		lambda r: r['RouteUID'],
		preserve_singletons=['Direction', 'Stops']
	)

	# Waypoints
	route_id = 'KHH1221'
	direction = 1
	WP = list(map(commons.inspect({'StopPosition': ('PositionLat', 'PositionLon')}), motc_routes[route_id]['Stops'][direction]))

	#print(list(map(commons.inspect('StopName'), motc_routes['KHH122']['Stops'][0])))

	# DEBUG
	WP = WP[0:3]
	# WP = WP[0:20]
	# WP = WP[-15:-11]

	#

	print("Locating nearest edges to waypoints...")

	def nearest_nodes(q, k=10) :

		# Find nearest nodes
		(dist, ind) = knn_tree.query(np.asarray(q).reshape(1, -1), k=k)

		# Get the in-graph node indices and flatten the arrays
		(dist, ind) = (dist.reshape(-1), [knn_ids[i] for i in ind.reshape(-1)])

		# Return a list of pairs (graph-node-id, distance-to-q) sorted by distance
		return list(zip(ind, dist))

	def nearest_edges(q, k=10) :
		# Get a number of closest nodes
		nn = [n for (n, d) in nearest_nodes(q, k=2*k+20)]
		# Get their incident edges
		ee = list(G.edges(nbunch=nn))
		# Append reverse edges
		ee = list(set(ee + [(b, a) for (a, b) in ee if G.has_edge(b, a)]))
		# Attach distance to q
		ee = [
			(e, dist_to_segment(q, (node_pos[e[0]], node_pos[e[1]]))[0])
			for e in ee
		]
		# Sort by distance
		ee = sorted(ee, key=(lambda ed : ed[1]))
		# Get the closest ones
		ee = ee[0:k]
		return ee

	# A cloud of edges for each waypoint
	dist_clouds = [dict(nearest_edges(wp)) for wp in WP]

	print("Connecting clouds...")

	def normalize_cloud(prob_cloud) :
		return {e : p / sum(prob_cloud.values()) for (e, p) in prob_cloud.items()}

	prob_clouds = [
		{
			e : 1
			for (e, dist) in cloud.items()
		}
		for cloud in dist_clouds
	]

	prob_clouds = [normalize_cloud(cloud) for cloud in prob_clouds]

	(miss_dist, total_len) = (None, None)

	# Intermediate edges -- initial condition
	ee = None

	# Partial shortest paths
	sps_way = dict()
	sps_len = dict()

	# Remaining clouds to optimize
	rcto = list(range(len(prob_clouds)))

	# Edge-to-length dictionary
	edge_length = nx.get_edge_attributes(G, 'len')

	# Plotting business

	import matplotlib.pyplot as plt

	fig : plt.Figure
	ax : plt.Axes
	(fig, ax) = plt.subplots()

	for (n, (y, x)) in enumerate(WP):
		ax.plot(x, y, 'bo')

	# Get the dimensions of the plot (again)
	(left, right, bottom, top) = ax.axis()

	# Compute a nicer aspect ratio if it is too narrow
	(w, h, phi) = (right - left, top - bottom, (1 + math.sqrt(5)) / 2)
	if (w < h / phi) : (left, right) = (((left + right) / 2 + s * h / phi / 2) for s in (-1, +1))
	if (h < w / phi) : (bottom, top) = (((bottom + top) / 2 + s * w / phi / 2) for s in (-1, +1))

	# Set new dimensions
	ax.axis([left, right, bottom, top])

	# Bounding box for the map
	bbox = (left, bottom, right, top)

	ax.autoscale(enable=False)

	token = open("../.credentials/UV/mapbox-token.txt", 'r').read()

	# Download the background map
	mapi = maps.get_map_by_bbox(bbox, token=token)

	plt.ion()
	plt.show()


	# Optimization loop
	while rcto :

		if ee is None :
			ee = [random.choices(list(pc.keys()), weights=list(pc.values()), k=1).pop() for pc in prob_clouds]

		for _ in range(23) :
			if not rcto : break

			try :

				# Choose a random edge cloud
				# nc = number of the edge cloud
				nc = random.choice(rcto)

				for _ in range(10) :

					# pc = edge weights in this cloud (modified below)
					pc = prob_clouds[nc]

					# Cloud edges with weights
					(ec, cw) = map(list, (pc.keys(), pc.values()))
					# exclude the currently selected edge
					(ec, cw) = zip(*[(e, p) for (e, p) in zip(ec, cw) if (e != ee[nc])])

					# Choose a candidate edge from the cloud (w/o the currently selected edge)
					ee[nc] = random.choices(ec, weights=cw, k=1).pop()
					# Current candidate edge
					ce = ee[nc]

					(old_miss_dist, old_total_len) = (miss_dist, total_len)

					# Criterion 1: Sum of distances of the selected edges to waypoints
					miss_dist = sum(dc[e] for (e, dc) in zip(ee, dist_clouds))

					# Criterion 2: The total length of the path
					total_len = sum(edge_length[e] for e in ee)
					path = [ee[0][0]] # First node of the first edge starts the complete path
					for (e, f) in zip(ee, ee[1:]) :
						(a, b) = (e[1], f[0])
						if (a, b) not in sps_way :
							sp = nx.shortest_path(G, source=a, target=b, weight='len')
							sps_way[(a, b)] = sp
							sps_len[(a, b)] = sum(edge_length[e] for e in zip(sp, sp[1:]))
						path += sps_way[(a, b)]
						total_len += sps_len[(a, b)]

					# Criterion 3: Accumulated turn angle
					# https://rosettacode.org/wiki/Angle_difference_between_two_bearings#Python
					# https://stackoverflow.com/a/16180796/3609568
					# http://gmc.yoyogames.com/index.php?showtopic=532420
					for (a, b, c) in zip(path, path[1:], path[2:]) :
						ba = angles.bear(*list(map(angles.d2r, [*b, *a])))
						bc = angles.bear(*list(map(angles.d2r, [*b, *c])))
						print(ba, bc)

					print("miss dist: {}, total len: {}".format(miss_dist, total_len))

					# If this edge has accumulated large weight
					# then consider this cloud "solved"

					if (prob_clouds[nc][ce] >= 0.99 * sum(prob_clouds[nc].values())) :
						# Remove this cloud number from the list of
						# remaining clouds to optimize
						rcto.remove(nc)
						break

					# Did we get any improvement in the criteria?

					def crit(md, tl) : return (md * 10) + tl

					# Relative improvement new/old
					rel = crit(miss_dist, total_len) / crit(old_miss_dist, old_total_len)

					# Re-weight the current edge in its cloud
					prob_clouds[nc][ce] *= (1.2 if (rel < 1) else 0.8)


				# # GPX
				#
				# # for cloud in clouds :
				# # 	for i in cloud.keys() :
				# # 		(y, x) = node_pos[i]
				# # 		gpx.waypoints.append(gpxpy.gpx.GPXWaypoint(latitude=y, longitude=x))
				# # 		ax.plot(x, y, 'bx')
				#
				# gpx = gpxpy.gpx.GPX()
				#
				# gpx_track = gpxpy.gpx.GPXTrack()
				# gpx.tracks.append(gpx_track)
				#
				# gpx_segment = gpxpy.gpx.GPXTrackSegment()
				# gpx_track.segments.append(gpx_segment)
				#
				# for i in path :
				# 	(p, q) = node_pos[i]
				# 	gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=p, longitude=q))
				#
				# with open("tmp.gpx", 'w') as f:
				# 	f.write(gpx.to_xml())

			except nx.NetworkXNoPath as e :
				prob_clouds[nc][ce] *= 0.9
				print("No path error", prob_clouds[nc])
				#ee = None
			except Exception as e :
				print(e)

			# PLOT

			# Clear the axes
			ax.cla()

			# Apply the background map
			ax.axis((left, right, bottom, top))
			img = ax.imshow(mapi, extent=(left, right, bottom, top), interpolation='quadric', zorder=-100)

			if path :
				(y, x) = zip(*[node_pos[i] for i in path])
				ax.plot(x, y, 'b--', linewidth=2, zorder=-50)

			for (n, (y, x)) in enumerate(WP) :
				c = 'm' #('g' if (n == nc) else 'r')
				ax.plot(x, y, 'o', c=c, markersize=4)

			for (nc, pc) in enumerate(prob_clouds) :
				m = max(pc.values())
				for (e, p) in pc.items() :
					(y, x) = zip(*[node_pos[i] for i in e])
					c = ('g' if (ee[nc] == e) else 'r')
					ax.plot(x, y, '-', c=c, linewidth=1, alpha=p/m, zorder=150)

			plt.draw()

			plt.show()
			plt.pause(0.5)

	plt.ioff()
	plt.show()

if (__name__ == "__main__") :
	foo()
