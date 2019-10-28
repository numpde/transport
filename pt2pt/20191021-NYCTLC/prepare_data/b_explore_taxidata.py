
# RA, 2019-10-24

import os
import math

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

#
import maps


PARAM = {
	'taxidata': "data/taxidata/sqlite/UV/db.db",

	'out_images_path': "exploration/",

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

	col_name = 'trip_distance/m'
	sql = F"SELECT [{col_name}] FROM [{table_name}]"
	trip_distance = query(sql)[col_name]

	# Convert to km
	trip_distance *= (1e-3)

	fig: plt.Figure
	ax1: plt.Axes
	(fig, ax1) = plt.subplots()
	trip_distance.hist(ax=ax1)
	ax1.set_yscale('log')
	ax1.set_xlabel('Trip distance, km')
	ax1.set_ylabel('Number of trips')
	ax1.set_title(F"Table: {table_name}")

	fn = os.path.join(PARAM['out_images_path'], F"{myname()}/{table_name}.png")
	fig.savefig(makedirs(fn), **PARAM['savefig_args'])


def trip_trajectories_initial(table_name):
	mpl.use("Agg")

	N = 10000

	cols = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]

	sql = F"SELECT {(', '.join(cols))} FROM [{table_name}] ORDER BY RANDOM() LIMIT {N}"
	df = query(sql)
	df = df.rename(columns=dict(zip(cols, ["lat0", "lon0", "lat1", "lon1"])))

	fig: plt.Figure
	ax1: plt.Axes
	(fig, ax1) = plt.subplots()
	ax1.tick_params(axis='both', which='both', labelsize='3')

	c = "b"
	if ("green" in table_name): c = "green"
	if ("yellow" in table_name): c = "orange"

	for (yy, xx) in zip(df[['lat0', 'lat1']].values, df[['lon0', 'lon1']].values):
		ax1.plot(xx, yy, c=c, ls='-', alpha=0.1, lw=0.1)

	ax1.axis("off")

	# Get the background map
	axis = ax1.axis()
	img_map = maps.get_map_by_bbox(maps.ax2mb(*axis))

	ax1.imshow(img_map, extent=axis, interpolation='quadric', zorder=-100)

	fn = os.path.join(PARAM['out_images_path'], F"{myname()}/{table_name}.png")
	fig.savefig(makedirs(fn), **PARAM['savefig_args'])


def pickup_hour_heatmap(table_name):
	mpl.use("Agg")

	col_name = 'pickup_datetime'
	sql = F"SELECT [{col_name}] FROM [{table_name}]"  # ORDER BY RANDOM() LIMIT 1000"
	pickup = pd.to_datetime(query(sql)[col_name])

	# Number of rides by weekday and hour
	df: pd.DataFrame
	df = pd.DataFrame({'d': pickup.dt.weekday, 'h': pickup.dt.hour})
	df = df.groupby(['d', 'h']).size().reset_index()
	df = df.pivot(index='d', columns='h', values=0)
	df = df.sort_index()

	# Average over the number of weekdays in the dataset
	df = df.div(pd.Series(Counter(d for (d, g) in groupby(pickup.sort_values().dt.weekday))), axis='index')

	fig: plt.Figure
	ax1: plt.Axes
	(fig, ax1) = plt.subplots()

	im = ax1.imshow(df, cmap=plt.get_cmap("Blues"), origin="upper")

	(xlim, ylim) = (ax1.get_xlim(), ax1.get_ylim())
	ax1.set_xticks(np.linspace(-0.5, 23.5, 25))
	ax1.set_xticklabels(range(0, 25))
	ax1.set_yticks(ax1.get_yticks(minor=False), minor=False)
	ax1.set_yticklabels([dict(enumerate(calendar.day_abbr)).get(int(t), "") for t in ax1.get_yticks(minor=False)])
	ax1.set_xlim(*xlim)
	ax1.set_ylim(*ylim)

	# cax = fig.add_axes([ax1.get_position().x1 + 0.01, ax1.get_position().y0, 0.02, ax1.get_position().height])
	# cbar = fig.colorbar(im, cax=cax)

	for i in range(df.shape[0]):
		for j in range(df.shape[1]):
			alignment = dict(ha="center", va="center")
			im.axes.text(j, i, int(round(df.loc[i, j])), fontsize=3, **alignment)

	fn = os.path.join(PARAM['out_images_path'], F"{myname()}/{table_name}.png")
	fig.savefig(makedirs(fn), **PARAM['savefig_args'])


def trip_speed_histogram(table_name):
	mpl.use("Agg")
	# logger.warning("TkAgg is used")

	sql = F"SELECT pickup_datetime as a, dropoff_datetime as b, [trip_distance/m] as m FROM [{table_name}]"
	# sql += "ORDER BY RANDOM() LIMIT 1000"  # DEBUG
	df: pd.DataFrame
	df = query(sql)
	df.a = pd.to_datetime(df.a)
	df.b = pd.to_datetime(df.b)
	# Duration in seconds
	df['s'] = (df.b - df.a).dt.total_seconds()
	# Hour of the day
	df['h'] = df.b.dt.hour
	# Forget pickup/dropoff times
	df = df.drop(columns=['a', 'b'])
	# Columns: distance-meters, duration-seconds, hour-of-the-day
	assert(all(df.columns == ['m', 's', 'h']))

	# Estimated average trip speed
	df['v'] = df.m / df.s

	(H, Min, Sec) = (1 / (60 * 60), 1 / 60, 1)

	# Omit low- and high-speed trips [m/s]
	MIN_TRIP_SPEED = 0.1  # m/s
	MAX_TRIP_SPEED = 15.5  # m/s
	df = df[df.v >= MIN_TRIP_SPEED]
	df = df[df.v <= MAX_TRIP_SPEED]

	# Omit long-duration trips
	MAX_TRIP_DURATION_H = 2  # hours
	df = df[df.s < MAX_TRIP_DURATION_H * (Sec / H)]

	fig: plt.Figure
	ax1: plt.Axes
	# Note: the default figsize is W x H = 8 x 6
	(fig, ax24) = plt.subplots(24, 1, figsize=(8, 12))

	xticks = list(range(int(math.ceil(df.v.max()))))

	for (h, hdf) in df.groupby(df['h']):
		c = plt.get_cmap("twilight_shifted")([h / 24])
		if ("green" in table_name): c = "green"
		if ("yello" in table_name): c = "orange"

		ax1 = ax24[h]
		ax1.hist(hdf.v, bins='fd', lw=2, density=True, histtype='step', color=c, zorder=10)
		ax1.set_xlim(0, max(df.v))
		ax1.set_yticks([])
		ax1.set_ylabel(F"{h}h", fontsize=6, rotation=90)
		ax1.set_xticks(xticks)
		ax1.set_xticklabels([])
		ax1.grid()

	ax1.set_xticklabels(xticks)
	ax1.set_xlabel("m/s")

	fn = os.path.join(PARAM['out_images_path'], F"{myname()}/{table_name}.png")
	fig.savefig(makedirs(fn), **PARAM['savefig_args'])


def running_number_of_trips(table_name):
	sql = F"SELECT pickup_datetime as a, dropoff_datetime as b FROM [{table_name}]"
	# sql += "ORDER BY RANDOM() LIMIT 1000"  # DEBUG
	df: pd.DataFrame
	df = query(sql).apply(pd.to_datetime, axis=0)

	logger.debug(df['b'].sort_values())
	raise NotImplementedError


def x():
	pass


def main():
	tables = {"green_tripdata_2016-05", "yellow_tripdata_2016-05"}

	for table_name in sorted(tables):
		trip_distance_histogram(table_name)
		trip_trajectories_initial(table_name)
		pickup_hour_heatmap(table_name)
		trip_speed_histogram(table_name)
		running_number_of_trips(table_name)
		pass


if __name__ == '__main__':
	main()

