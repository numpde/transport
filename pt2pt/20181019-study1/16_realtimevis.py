#!/usr/bin/python3

# RA, 2018-10-31

## ================== IMPORTS :

import commons

import time
import json
import glob
import inspect
from itertools import chain


## ==================== NOTES :

pass


## ==================== INPUT :

IFILE = {
	'response' : "OUTPUT/12/Kaohsiung/UV/{d}/{t}.json",

	'route-stops' : "ORIGINALS/MOTC/Kaohsiung/CityBusApi_StopOfRoute/data.json",
}


## =================== OUTPUT :

OFILE = {
}


## ==================== PARAM :

PARAM = {
}

## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))


## ===================== WORK :

pass

## ===================== PLAY :

def test_extract_runs() :

	response_files = sorted(glob.glob(IFILE['response'].format(d="20181101", t="1*")))
	time.sleep(1)

	# Extracted bus runs
	all_runs = [ ]
	cur_runs = { }

	k = 0

	for fn in response_files :

		print(fn)

		J = commons.zipjson_load(fn)

		if not J : continue

		k += 1

		B = commons.index_dicts_by_key(
			J,
			lambda b : "{}/{}/{}/{}/{}".format(b['PlateNumb'], b['SubRouteUID'], b['Direction'], b['DutyStatus'], b.get('BusStatus', '?'))
		)

		for (i, b) in B.items() :
			if not (i in cur_runs) :
				cur_runs[i] = ([{}] if k else [])

			cur_runs[i].append(b)

		for i in (set(cur_runs.keys()) - set(B.keys())) :
			# Close run and move it to history
			all_runs.append(cur_runs.pop(i) + [{}])

			# TODO: clean up tail


	import matplotlib.pyplot as plt

	plt.ion()
	plt.show()

	def plot_run(run) :
		p = [(bp['PositionLon'], bp['PositionLat']) for bp in [b['BusPosition'] for b in run]]
		if (len(p) <= 1) : return
		(x, y) = zip(*p)
		plt.plot(x, y, '-', linewidth=1)

	for run in all_runs :
		# If the complete run is captured, it is delimited by empty dicts
		if run[0] : continue
		if run[-1]: continue
		run = run[1:-1]
		if (len(run) <= 2) : continue
		if not (run[0].get('BusStatus') == 0) : continue
		if not (run[0].get('DutyStatus') == 0) : continue
		print(run)
		plot_run(run)
		plt.draw()
		plt.pause(0.1)


# Small visualization of the bus record
def vis1() :

	response_files = sorted(glob.glob(IFILE['response'].format(d="20181101", t="*")))
	time.sleep(1)

	import matplotlib.pyplot as plt

	plt.ion()
	plt.show()

	B = { }

	route_uids = ['KHH122'] #, 'KHH1221', 'KHH882']

	# Indicate bus stops

	routes = commons.index_dicts_by_key(commons.zipjson_load(IFILE['route-stops']), (lambda r : r['RouteUID']))
	routes = { i : routes[i] for i in route_uids }

	for (i, route) in routes.items() :
		# Got a list after applying "index_dicts_by_key"
		assert(type(route['Stops']) is list)
		for (dir, stops) in zip(route['Direction'], route['Stops']) :
			for stop in stops :
				n = stop['StopName']
				p = stop['StopPosition']
				(y, x) = (p['PositionLat'], p['PositionLon'])
				plt.scatter(x, y, c='b', marker={0: 'x', 1: '+'}[dir] )

	# Show bus movement

	# Extracted bus runs
	all_runs = [ ]
	cur_runs = { }

	def plot_run(run) :
		p = [(bp['PositionLon'], bp['PositionLat']) for bp in [b['BusPosition'] for b in run]]
		if (len(p) <= 1) : return
		(x, y) = zip(*p)
		plt.plot(x, y, '-', linewidth=1)

	for fn in response_files :

		J = commons.zipjson_load(fn)

		# Filter down to one route
		J = [j for j in J if (j['SubRouteUID'] in route_uids)]

		# Sort by plate number
		J = sorted(J, key=(lambda j: j['PlateNumb']))

		# Filter by direction
		J = [j for j in J if (j['Direction'] == 0)]

		if not J : continue

		try :
			# The plate number should be unique
			assert(len(J) == len(set(j['PlateNumb'] for j in J)))
		except AssertionError :
			# It is not always the case!
			# TODO
			continue

		B_before = B

		# Index bus info by plate number
		B = { b['PlateNumb'] : b for b in J }

		for (pn, b) in B.items() :
			if not (pn in cur_runs) :
				# If a new plate number: start new current run
				cur_runs[pn] = [b]
			else :
				# Append to existing current run
				cur_runs[pn].append(b)

		# If a plate number of a current run disappears
		for pn in (set(cur_runs.keys()) - set(B.keys())) :
			# Move it to history
			# TODO: clean up tail
			all_runs.append(cur_runs.pop(pn))

		# # Position by plate number
		# def P(B) :
		# 	return { pn : (b['BusPosition']['PositionLon'], b['BusPosition']['PositionLat']) for (pn, b) in B.items() }
		#
		# plt.clf()
		#
		# for run in cur_runs.values() :
		# 	plot_run(run)
		#
		# # Style
		# s = {
		# 	pn : {0: 'r', 90: 'g', 98: 'm', 99: 'k'}[int(B[pn]['BusStatus'])]
		# 	for pn in P(B).keys()
		# }
		#
		# # for pn in set.intersection(set(P(B).keys()), set(P(B_before).keys())) :
		# # 	plt.plot(*zip(P(B_before)[pn], P(B)[pn]), '-' + s[pn], linewidth=0.1)
		#
		# (x, y) = zip(*P(B).values())
		# h = plt.scatter(x, y, c=list(s[pn] for pn in P(B).keys()))
		#
		# plt.draw()
		# plt.pause(0.1)
		#
		# h.remove()

	plt.clf()

	for run in all_runs :
		plot_run(run)

	plt.draw()
	plt.show()
	plt.pause(0.1)

	time.sleep(5)
	input("Please press ENTER")


## ================== OPTIONS :

OPTIONS = {
	'VIS1' : vis1,
	'TEST_EXTRACT' : test_extract_runs,
}

## ==================== ENTRY :

if (__name__ == "__main__") :

	if not commons.parse_options(OPTIONS) :

		print("Please specify option via command line:", *OPTIONS.keys())

