
# RA, 2019-10-24

import os
from sqlite3 import connect
import pandas as pd

import inspect

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Agg")

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


def main():
	tables = {"green_tripdata_2016-05", "yellow_tripdata_2016-05"}

	for table_name in tables:
		trip_distance_histogram(table_name)
		trip_trajectories_plot(table_name)


if __name__ == '__main__':
	main()

