#!/usr/bin/python3

# RA, 2018-10-21

## ================== IMPORTS :

from helpers import commons, transit, maps

import datetime as dt
import inspect
import pytz

import networkx as nx


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
	'' : "",
}


## ================== PARAM 2 :


## ====================== AUX :

def ll2xy(latlon) :
	return (latlon[1], latlon[0])

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))


## ==================== TESTS :

def test1() :

	import matplotlib.pyplot as plt
	plt.ion()

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


	tr = transit.Transit(commons.ls(IFILE['timetable_json'].format(routeid="*", dir="*")))

	if True :
		t0 = PARAM['TZ'].localize(dt.datetime(year=2018, month=11, day=6, hour=13, minute=15))
		origin = 'KHH4439'
		legs = tr.connect(transit.Loc(t=t0, desc=origin), callback=plot_callback)

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

		print("Finding a route from {} to {} at {}".format(stop_a, stop_z, t0))

		legs = tr.connect(transit.Loc(t=t0, desc=stop_a), transit.Loc(desc=stop_z)) #, callback=plot_callback)

		print(legs[0], legs[-1])

		for leg in legs :
			(t0, t1) = (pytz.utc.localize(t).astimezone(tz=PARAM['TZ']) for t in (leg.P.t, leg.Q.t))
			print("{}-{} : {} {}".format(t0.strftime("%Y-%m-%d %H:%M"), t1.strftime("%H:%M (%Z)"), leg.mode, leg.desc))


	print("Done.")

	plt.ioff()
	plt.show()



## ===================== WORK :


## ==================== ENTRY :

if (__name__ == "__main__") :
	test1()

