
# RA, 2019-10-24

import os
from sqlite3 import connect

import calendar

import numpy as np
import pandas as pd

import inspect

from itertools import groupby
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt

import logging as logger
logger.basicConfig(level=logger.DEBUG, format="%(levelname)-8s [%(asctime)s] : %(message)s", datefmt="%Y%m%d %H:%M:%S %Z")
logger.getLogger('matplotlib').setLevel(logger.WARNING)
logger.getLogger('PIL').setLevel(logger.WARNING)

import percache
cache = percache.Cache("/tmp/percache_" + os.path.basename(__file__), livesync=True)

import maps


PARAM = {
	'taxidata': "data/taxidata/sqlite/UV/db.db",

	'out_images_path': "data/taxidata/exploration/",

	'savefig_args': dict(bbox_inches='tight', pad_inches=0, dpi=300),
}


def makedirs(path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	return path


# Return caller's function name
def myname():
	return inspect.currentframe().f_back.f_code.co_name


def query(sql) -> pd.DataFrame:
	logger.debug(F"Query: {sql}")
	with connect(PARAM['taxidata']) as con:
		return pd.read_sql_query(sql, con)


def trip_distance_histogram(table_name):
	mpl.use("Agg")

	col_name = 'trip_distance'
	sql = F"SELECT [{col_name}] FROM [{table_name}]"
	trip_distance = query(sql)[col_name]

	fig: plt.Figure
	ax1: plt.Axes
	(fig, ax1) = plt.subplots()
	trip_distance.hist(ax=ax1)
	ax1.set_yscale('log')
	ax1.set_xlabel('Trip distance, miles')
	ax1.set_ylabel('Number of trips')
	ax1.set_title(F"Table: {table_name}")

	fn = os.path.join(PARAM['out_images_path'], F"{myname()}/{table_name}.png")
	fig.savefig(makedirs(fn), **PARAM['savefig_args'])


def trip_trajectories_plot(table_name):
	mpl.use("Agg")

	cols = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]

	sql = F"SELECT {(', '.join(cols))} FROM [{table_name}] ORDER BY RANDOM() LIMIT 10000"
	df = query(sql)
	df = df.rename(columns=dict(zip(cols, ["lat0", "lon0", "lat1", "lon1"])))

	fig: plt.Figure
	ax1: plt.Axes
	(fig, ax1) = plt.subplots()
	ax1.tick_params(axis='both', which='both', labelsize='3')

	for (yy, xx) in zip(df[['lat0', 'lat1']].values, df[['lon0', 'lon1']].values):
		ax1.plot(xx, yy, 'b-', alpha=0.1, lw=0.1)

	# Get the background map
	axis = ax1.axis()
	img_map = maps.get_map_by_bbox(maps.ax2mb(*axis))

	ax1.imshow(img_map, extent=axis, interpolation='quadric', zorder=-100)

	fn = os.path.join(PARAM['out_images_path'], F"{myname()}/{table_name}.png")
	fig.savefig(makedirs(fn), **PARAM['savefig_args'])


def trip_hour_histogram(table_name):
	mpl.use("TkAgg")

	col_name = 'pickup_datetime'
	sql = F"SELECT [{col_name}] FROM [{table_name}]" # ORDER BY RANDOM() LIMIT 100000"
	pickup = pd.to_datetime(query(sql)[col_name])
	df: pd.DataFrame
	df = pd.DataFrame({'d': pickup.dt.weekday, 'h': pickup.dt.hour})
	day_count = pd.Series(Counter(d for (d, g) in groupby(pickup.sort_values().dt.weekday)))
	df = df.groupby(['d', 'h']).size().reset_index()
	df = df.pivot(index='d', columns='h', values=0)
	# Average number of rides initiated
	df = df.div(day_count, axis='index')
	df = df.sort_index()

	fig: plt.Figure
	ax1: plt.Axes
	(fig, ax1) = plt.subplots()
	ax1.imshow(df, cmap=plt.get_cmap("Blues"), origin="upper")

	(xlim, ylim) = (ax1.get_xlim(), ax1.get_ylim())
	ax1.set_xticks(np.linspace(-0.5, 23.5, 25))
	ax1.set_xticklabels(range(0, 25))
	ax1.set_yticks(ax1.get_yticks(minor=False), minor=False)
	ax1.set_yticklabels([dict(enumerate(calendar.day_abbr)).get(int(t), "") for t in ax1.get_yticks(minor=False)])
	ax1.set_xlim(*xlim)
	ax1.set_ylim(*ylim)

	fn = os.path.join(PARAM['out_images_path'], F"{myname()}/{table_name}.png")
	fig.savefig(makedirs(fn), **PARAM['savefig_args'])


def main():
	tables = {"green_tripdata_2016-05", "yellow_tripdata_2016-05"}

	for table_name in sorted(tables):
		# trip_distance_histogram(table_name)
		# trip_trajectories_plot(table_name)
		trip_hour_histogram(table_name)


if __name__ == '__main__':
	main()

