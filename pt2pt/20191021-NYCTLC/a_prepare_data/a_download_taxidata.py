
# RA, 2019-10-21

# Download some NYC TLC taxi data


import os
import json

from urllib.request import urlretrieve as wget

import pandas as pd
import numpy as np

from sqlite3 import connect

from multiprocessing import Pool

from shapely import geometry
from shapely.prepared import prep as prep_geometry


PARAM = {
	'datapath': "data/taxidata/",

	'bounding_multipolygon_src': "data/osm/manhattan/manhattan.geojson",

	'min_trip_distance_miles': 0.1,
	'max_trip_distance_miles': 30,
}

PARAM.update(
	bounding_shape=max(
		(p for mp in geometry.shape(json.load(open(PARAM['bounding_multipolygon'], 'r'))) for p in mp),
		key=(lambda p: p.length)
	)
)

PARAM.update(
	transfers=[
		{
			# Sources
			'urls': {
				# NYC TLC data before 2017 provides lat/lon pairs
				"https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-05.csv",
				"https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2016-05.csv",
			},
			# Destination relative to this module
			'path': os.path.join(PARAM['datapath'], "monthly/UV/"),
		},
		{
			# Sources
			'urls': {
				"https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf",
				"https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_green.pdf",
			},
			# Destination
			'path': os.path.join(PARAM['datapath'], "monthly/meta/"),
		},
	]
)

database = os.path.join(PARAM['datapath'], "sqlite/UV/db.db")
os.makedirs(os.path.dirname(database), exist_ok=True)


def src_trg_tbl():
	for transfer in PARAM['transfers']:
		(urls, path) = (transfer['urls'], transfer['path'])
		for url in urls:
			trg = os.path.join(os.path.dirname(__file__), path, os.path.basename(url))
			yield (url, trg, os.path.splitext(os.path.basename(url))[0])


def data_clean(df: pd.DataFrame) -> pd.DataFrame:
	# Lowercase column names
	df.columns = map(str.strip, map(str.lower, df.columns))

	# Remove prefixes from pickup/dropoff datetimes
	for e in ["pickup_datetime", "dropoff_datetime"]:
		df.columns = map(lambda c: (e if c.endswith(e) else c), df.columns)

	# Omit rows with bogus lat/lon entries
	for spot in ["pickup", "dropoff"]:
		for coor in ["latitude", "longitude"]:
			df = df[df[spot + "_" + coor] != 0]

	# Omit rows with small/large trip distance (in miles)
	df = df[df['trip_distance'] >= PARAM['min_trip_distance_miles']]
	df = df[df['trip_distance'] <= PARAM['max_trip_distance_miles']]

	# Convert travel distance to meters
	METERS_PER_MILE = 1609.34
	df['trip_distance'] *= METERS_PER_MILE
	df = df.rename(columns={"trip_distance": "trip_distance/m"})

	return df


def data_in_polygon(df: pd.DataFrame, selector) -> pd.DataFrame:
	# Omit rows with pickup/dropoff outside of "area"
	for spot in ["pickup", "dropoff"]:
		cols = [F"{spot}_longitude", F"{spot}_latitude"]
		has = list(map(selector, map(geometry.Point, zip(*df[cols].values.T))))
		df = df.loc[has, :]
	return df


def download():
	for (url, trg, __) in src_trg_tbl():
		print(url, "=>", os.path.relpath(trg, os.path.dirname(__file__)))
		os.makedirs(os.path.dirname(trg), exist_ok=True)
		if os.path.isfile(trg):
			print("File exists -- skipping")
		else:
			print("Downloading -- please wait")
			wget(url, trg)


def tosqlite():
	for (__, file, table) in src_trg_tbl():
		if not file.endswith(("zip", "csv")):
			continue

		print("{} => {}".format(file, table))

		# Selector based on a region (e.g. Manhattan)
		# https://shapely.readthedocs.io/en/stable/manual.html#object.contains
		selector = prep_geometry(PARAM['bounding_shape']).contains

		with connect(database) as con:
			con.cursor().execute("DROP TABLE IF EXISTS [{}]".format(table))

			for df in pd.read_csv(file, chunksize=(1024 * 1024)):

				df = data_clean(df)
				df = data_in_polygon(df, selector)

				# Write to database
				df.to_sql(name=table, con=con, if_exists='append', index=False)


if __name__ == "__main__":
	download()
	tosqlite()

