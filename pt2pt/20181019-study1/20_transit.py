#!/usr/bin/python3

# RA, 2018-10-21

## ================== IMPORTS :

import datetime as dt
import inspect
import pytz
import json
import uuid

import numpy as np
import networkx as nx

import io

from helpers import commons, transit, maps, graph

## ==================== NOTES :

pass




## ================== PARAM 1 :

PARAM = {
	'city' : "Kaohsiung",
	'scenario' : "Kaohsiung/20181105-20181111",
	'TZ' : pytz.timezone('Asia/Taipei'),
}


## ==================== INPUT :

IFILE = {
	#'OSM-pickled' : "OUTPUT/02/UV/kaohsiung.pkl",

	# Will be loaded from timetable files:
	#'MOTC_routes' : "OUTPUT/00/ORIGINAL_MOTC/{city}/CityBusApi_StopOfRoute.json",
	#'MOTC_stops'  : "OUTPUT/00/ORIGINAL_MOTC/{city}/CityBusApi_Stop.json",

	#'MOTC_shapes' : "OUTPUT/00/ORIGINAL_MOTC/{city}/CityBusApi_Shape.json",

	'timetable_json' : "OUTPUT/17/timetable/{scenario}/json/{{routeid}}-{{dir}}.json",
}

for (k, s) in IFILE.items() : IFILE[k] = s.format(**PARAM)


## =================== OUTPUT :

OFILE = {
	'transit_map' : "OUTPUT/20/transit_map/v1/{uuid}.{ext}",
}

commons.makedirs(OFILE)


## ================== PARAM 2 :

PARAM.update({
	'mapbox' : {
		'token' : commons.token_for('mapbox'),
		'cachedir' : "helpers/maps_cache/UV/",
	},

})

## ====================== AUX :

def ll2xy(latlon) :
	return (latlon[1], latlon[0])

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))

# points is a dictionary (lat, lon) --> data
# returns a pairs (bbox, point dict)
def boxify(points: dict, maxinbox=5) :

	if not points : return

	get_lat = (lambda p: p[0])
	get_lon = (lambda p: p[1])

	(lat, lon) = zip(*points.keys())

	bbox = (min(lon), min(lat), max(lon), max(lat))

	if (len(points) <= maxinbox) :
		yield (bbox, points)
		return

	if ((max(lat) - min(lat)) >= (max(lon) - min(lon))) :
		# partition along lat
		first = (lambda p, m=((max(lat) + min(lat)) / 2) : (get_lat(p) < m))
	else :
		# partition along lon
		first = (lambda p, m=((max(lon) + min(lon)) / 2) : (get_lon(p) < m))

	yield from boxify({ p: d for (p, d) in points.items() if     first(p) }, maxinbox)
	yield from boxify({ p: d for (p, d) in points.items() if not first(p) }, maxinbox)


## ==================== TESTS :

def test1() :

	import matplotlib.pyplot as plt
	plt.ion()

	tr = transit.Transit(commons.ls(IFILE['timetable_json'].format(routeid="*", dir="*")))

	if True :
		t0 = PARAM['TZ'].localize(dt.datetime(year=2018, month=11, day=6, hour=13, minute=15))
		origin = 'KHH4439'
		legs = tr.connect(transit.Loc(t=t0, desc=origin), callback=plot_callback)


	def plot_callback(result) :
		if (result['status'] == "zero") :
			return

		if (result['status'] == "init") :
			(fig, ax) = plt.subplots()
			result['fig'] = fig
			result['ax'] = ax
			return

		if (result['status'] == "opti") :
			if (result.get('nfu', dt.datetime.min) > dt.datetime.now()) :
				return

		astar_initial = result['astar_initial']
		astar_targets = result['astar_targets']
		astar_openset = result['astar_openset']
		astar_graph = result['astar_graph']
		routes = result['routes']
		stop_pos = result['stop_pos']

		ax: plt.Axes
		ax = result['ax']

		ax.cla()

		nx.draw_networkx_edges(astar_graph, ax=ax, edgelist=[(a, b) for (a, b, d) in astar_graph.edges.data('leg') if (d.mode == transit.Mode.walk)], pos=nx.get_node_attributes(astar_graph, 'pos'), edge_color='g', arrowsize=5, node_size=0)
		nx.draw_networkx_edges(astar_graph, ax=ax, edgelist=[(a, b) for (a, b, d) in astar_graph.edges.data('leg') if (d.mode == transit.Mode.bus)], pos=nx.get_node_attributes(astar_graph, 'pos'), edge_color='b', arrowsize=5, node_size=0)

		if astar_initial :
			ax.plot(*zip(*[ll2xy(P.x) for P in astar_initial.values()]), 'go')
		if astar_targets :
			ax.plot(*zip(*[ll2xy(P.x) for P in astar_targets.values()]), 'ro')
		if astar_openset :
			ax.plot(*zip(*[ll2xy(O.x) for O in astar_openset.values()]), 'kx')

		if not astar_openset :
			try :
				for leg in legs :
					(y, x) = zip(leg.P.x, leg.Q.x)
					ax.plot(x, y, 'y-', alpha=0.3, linewidth=8, zorder=100)
			except :
				pass

		a = ax.axis()
		for route in routes.values() :
			(y, x) = zip(*(stop_pos[stop['StopUID']] for stop in route['Stops']))
			ax.plot(x, y, 'm-', alpha=0.1)
		ax.axis(a)

		plt.pause(0.1)

		result['nfu'] = dt.datetime.now() + dt.timedelta(seconds=2)


	if False :
		t0 = PARAM['TZ'].localize(dt.datetime(year=2018, month=11, day=6, hour=13, minute=15))
		#print("Departure time: {}".format(t0.strftime("%Y-%m-%d %H:%M (%Z)")))


		#
		#t0 = t0.astimezone(dt.timezone.utc).replace(tzinfo=None)


		#bb = BusstopBusser(routes, stop_pos)

		# print(bb.where_can_i_go('KHH308', t))
		# exit(39)

		# print("Routes passing through {}: {}".format(A, bb.routes_from[A]))
		# print(bb.routes[('KHH100', 0)]['Stops'])
		# exit(39)

		#bw = BusstopWalker(stop_pos)
		# print(bw.where_can_i_go('KHH308', t))
		# bw.where_can_i_go('KHH380', t)

		# Random from-to pair
		(stop_a, stop_z) = commons.random_subset(tr.stop_pos.keys(), k=2)
		# Relatively short route, many buses
		(stop_a, stop_z) = ('KHH4439', 'KHH4370')
		# # Long search, retakes same busroute
		# (stop_a, stop_z) = ('KHH3820', 'KHH4484')

		commons.logger.info("Finding a route from {} to {} at {}".format(stop_a, stop_z, t0))

		legs = tr.connect(transit.Loc(t=t0, desc=stop_a), transit.Loc(desc=stop_z)) #, callback=plot_callback)

		commons.logger.debug("First/last legs: {}/{}".format(legs[0], legs[-1]))

		for leg in legs :
			(t0, t1) = (pytz.utc.localize(t).astimezone(tz=PARAM['TZ']) for t in (leg.P.t, leg.Q.t))
			commons.logger.info("{}-{} : {} {}".format(t0.strftime("%Y-%m-%d %H:%M"), t1.strftime("%H:%M (%Z)"), leg.mode, leg.desc))


	commons.logger.info("Done.")

	plt.ioff()
	plt.show()



## =================== SLAVES :

def map_transit_from(t: dt.datetime, x) :

	def tr_callback(result) :
		if (result['status'] in {"zero", "init"}) :
			return

		# Next callback update
		if (result['status'] == "opti") :
			if (result.get('ncu', dt.datetime.min) > dt.datetime.now()) :
				return
			else :
				result['ncu'] = dt.datetime.now() + dt.timedelta(seconds=10)

		g: nx.DiGraph
		g = result['astar_graph']
		J = {
			'origin' : {
				'x' : x,
				't' : t.isoformat(),
			},
			'gohere' : [
				{
					'x' : g.nodes[n]['loc'].x,
					's' : (g.nodes[n]['loc'].t - t.astimezone(dt.timezone.utc).replace(tzinfo=None)).total_seconds(),
					'o' : (g.nodes[next(iter(g.pred[n]))]['loc'].x if g.pred[n] else None),
				}
				for n in list(g.nodes)
			],
		}

		# Preserve the UUID and the filename between callbacks
		fn = OFILE['transit_map'].format(uuid=result.setdefault('file_uuid', uuid.uuid4().hex), ext="json")

		with open(fn, 'w') as fd :
			json.dump(J, fd)

		commons.logger.info("Number of locations mapped is {}".format(g.number_of_nodes()))

	def keep_ttfile(fn) :
		J = commons.zipjson_load(fn)

		# "Inner" Kaohsiung
		bbox = (120.2593, 22.5828, 120.3935, 22.6886)
		(left, bottom, right, top) = bbox

		(lat, lon) = map(np.asarray, zip(*map(commons.inspect({'StopPosition' : ('PositionLat', 'PositionLon')}), J['route']['Stops'])))
		return all(map(all, [bottom <= lat, lat <= top, left <= lon, lon <= right]))

	# A* INITIALIZE
	commons.logger.info("Initializing transit...")
	tr = transit.Transit(filter(keep_ttfile, commons.ls(IFILE['timetable_json'].format(routeid="*", dir="*"))))

	# A* COMPUTE
	commons.logger.info("Computing transit from {} at {}...".format(x, t))
	tr.connect(transit.Loc(t=t, x=x), callback=tr_callback)


def make_transit_img(J) -> bytes :
	import matplotlib as mpl
	mpl.use('Agg')

	import matplotlib.pyplot as plt

	ax: plt.Axes
	(fig, ax) = plt.subplots()

	origin = {
		'x': J['origin']['x'],
		't' : J['origin']['t'],
		'desc' : J['origin'].get('desc'),
	}

	# Location --> Transit time in minutes ; keep track of duplicates
	gohere = commons.index_dicts_by_key(J['gohere'], key_func=(lambda __ : tuple(__['x'])), collapse_repetitive=False)
	# Keep only the minimal reach time, convert to minutes
	gohere = { x : (min(attr['s']) / 60) for (x, attr) in gohere.items() }

	# # Cut-off (and normalize)
	T = 60 # Minutes
	# gohere = { p : s for (p, s) in gohere.items() if (s <= T) }

	# Reindex datapoints by (x, y) pairs
	contour_pts = dict(zip(map(ll2xy, gohere.keys()), gohere.values()))

	#boxes = dict(boxify(gohere, maxinbox=10))

	# "Inner" Kaohsiung
	bbox = (120.2593, 22.5828, 120.3935, 22.6886)

	# Set plot view to the bbox
	ax.axis(maps.mb2ax(*bbox))
	ax.autoscale(enable=False)

	ax.tick_params(axis='both', which='both', labelsize='xx-small')

	try :
		background_map = maps.get_map_by_bbox(bbox, style=maps.MapBoxStyle.light, **PARAM['mapbox'])
		ax.imshow(background_map, interpolation='quadric', extent=maps.mb2ax(*bbox), zorder=-100)
	except Exception as e :
		commons.logger.warning("No background map ({})".format(e))

	ax.plot(*ll2xy(origin['x']), 'gx')

	# Show all datapoints
	ax.scatter(*zip(*contour_pts), marker='o', c='k', s=0.1, lw=0, edgecolor='none')

	# # Hack! for corners
	# for (x, y) in product(ax.axis()[:2], ax.axis()[2:]) :
	# 	contour_pts[(x, y)] = max(gohere.values())


	cmap = plt.get_cmap('Purples')
	cmap.set_over('k')

	# https://stackoverflow.com/questions/37327308/add-alpha-to-an-existing-matplotlib-colormap
	from matplotlib.colors import ListedColormap
	cmap = ListedColormap(np.vstack([cmap(np.arange(cmap.N))[:, 0:3].T, np.linspace(0, 0.5, cmap.N)]).T)

	(x, y) = zip(*contour_pts)
	levels = list(range(0, T, 5))
	c = ax.tricontourf(x, y, list(contour_pts.values()), levels=levels, zorder=100, cmap=cmap, extent=maps.mb2ax(*bbox), extend='max')


	cbar = fig.colorbar(c)
	cbar.ax.tick_params(labelsize='xx-small')

	# import matplotlib.patches as patches
	# for (bb, gohere_part) in boxes.items() :
	# 	#ax.add_patch(patches.Rectangle(bb[0:2], bb[2]-bb[0], bb[3]-bb[1], linewidth=0.5, edgecolor='k', facecolor='none'))
	# 	for (latlon, s) in list(gohere_part.items()) :
	# 		ax.plot(*ll2xy(latlon), 'o', c=plt.get_cmap('Purples')(s), markersize=3)

	buffer = io.BytesIO()
	fig.savefig(buffer, bbox_inches='tight', pad_inches=0, dpi=300)

	buffer.seek(0)

	return buffer.read()


## =================== MASTER :

def map_transit() :
	t = PARAM['TZ'].localize(dt.datetime(year=2018, month=11, day=6, hour=13, minute=15))
	# Unknown location
	x = (22.63322, 120.33468)
	# HonDo
	x = (22.63121, 120.32742)
	map_transit_from(t=t, x=x)


def img_transit() :

	for fn in commons.ls(OFILE['transit_map'].format(uuid="*", ext="json")) :
		with open(commons.reformat(OFILE['transit_map'], fn, {'ext': "png"}), 'wb') as fd :
			J = commons.zipjson_load(fn)
			fd.write(make_transit_img(J))


## ==================== DEBUG :

def debug_compare_two() :
	uuids = ["16b767f12ac841fea47ad9b735df1504", "69e47ef6a81a4a3aae0529b8b974896b"]
	(J1, J2) = (commons.zipjson_load(OFILE['transit_map'].format(uuid=uuid, ext="json")) for uuid in uuids)

	o = tuple(J1['origin']['x'])
	assert (J1['origin'] == J2['origin'])

	(H1, H2) = ({}, {})
	(O1, O2) = ({}, {})
	for (J, H, O) in zip([J1, J2], [H1, H2], [O1, O2]) :
		# Location --> Transit time in minutes ; keep track of duplicates
		J['gohere'] = commons.index_dicts_by_key(J['gohere'], key_func=(lambda __ : tuple(__['x'])), collapse_repetitive=False)
		# Keep the *time* field
		H.update({ x : attr['s'] for (x, attr) in J['gohere'].items() })
		# Keep the *origin* field
		O.update({ x : attr['o'] for (x, attr) in J['gohere'].items() })

	# The two datasets cover the same geo-locations
	assert (set(H1) == set(H2))

	X = sorted([x for x in H1 if (set(H1[x]) != set(H2[x]))], key=(lambda x : sum(H1[x]) + sum(H2[x])))
	# commons.logger.debug("Earliest differing location: {}".format(X[0]))

	for x in X[0:4] :

		g1 = nx.DiGraph()
		g2 = nx.DiGraph()

		def retrace(O, g, x) :
			for o in O[x] :
				if o :
					o = tuple(o)
					if not g.has_edge(o, x) :
						g.add_edge(o, x)
						retrace(O, g, o)
			g.nodes[x]['pos'] = ll2xy(x)

		retrace(O1, g1, x)
		retrace(O2, g2, x)

		commons.logger.debug("Graph 1: {}".format(g1.nodes))
		commons.logger.debug("Graph 2: {}".format(g2.nodes))



		import matplotlib as mpl
		mpl.use("TkAgg")

		import matplotlib.pyplot as plt
		(fig, ax) = plt.subplots()

		# # "Inner" Kaohsiung
		# bbox = (120.2593, 22.5828, 120.3935, 22.6886)
		# # Set plot view to the bbox
		# ax.axis(maps.mb2ax(*bbox))
		# ax.autoscale(enable=False)

		nx.draw_networkx(g1, pos=nx.get_node_attributes(g1, 'pos'), edge_color='b', node_size=1, with_labels=False)
		nx.draw_networkx(g2, pos=nx.get_node_attributes(g2, 'pos'), edge_color='g', node_size=1, with_labels=False)

		# ax.plot(*o, 'gx')
		# ax.scatter(*zip(*nx.get_node_attributes(g1, 'pos').values()), marker='o', c='k', s=0.1, lw=0, edgecolor='none')

		plt.show()


## ================== OPTIONS :r

OPTIONS = {
	'MAP' : map_transit,
	'IMG' : img_transit,

	'DBG' : debug_compare_two,
}


## ==================== ENTRY :

if (__name__ == "__main__") :
	commons.parse_options(OPTIONS)
