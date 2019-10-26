
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

data_path = "data/taxidata/"

manhattan: geometry.Polygon
manhattan = max(
	(
		p
		for mp in geometry.shape(json.load(open("data/osm/manhattan/manhattan.geojson", 'r')))
		for p in mp
	),
	key=(lambda p: p.length)
)

transfers = [
	{
		# Sources
		'urls': {
			# NYC TLC data before 2017 provides lat/lon pairs
			"https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-05.csv",
			"https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2016-05.csv",
		},
		# Destination relative to this module
		'path': os.path.join(data_path, "monthly/UV/"),
	},
	{
		# Sources
		'urls': {
			"https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf",
			"https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_green.pdf",
		},
		# Destination
		'path': os.path.join(data_path, "monthly/meta/"),
	},
]

database = os.path.join(data_path, "sqlite/UV/db.db")
os.makedirs(os.path.dirname(database), exist_ok=True)

# bboxfile = os.path.join(data_path, "bbox.json")


def src_trg_tbl():
	for transfer in transfers:
		(urls, path) = (transfer['urls'], transfer['path'])
		for url in urls:
			trg = os.path.join(os.path.dirname(__file__), path, os.path.basename(url))
			yield (url, trg, os.path.splitext(os.path.basename(url))[0])


def data_clean(df: pd.DataFrame) -> pd.DataFrame:
	# Omit rows with bogus lat/lon entries
	for spot in ["pickup", "dropoff"]:
		for coor in ["latitude", "longitude"]:
			df = df[df[spot + "_" + coor] != 0]

	# Omit rows with large trip distance (in miles)
	MAX_TRIP_DISTANCE_MILES = 50
	df = df[df['trip_distance'] <= MAX_TRIP_DISTANCE_MILES]

	return df


def data_in_polygon(df: pd.DataFrame, area: geometry.Polygon) -> pd.DataFrame:

	# Omit rows with pickup/dropoff outside of "area"
	for spot in ["pickup", "dropoff"]:
		cols = [F"{spot}_longitude", F"{spot}_latitude"]
		has = list(map(prep_geometry(area).contains, map(geometry.Point, zip(*df[cols].values.T))))
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

		with connect(database) as con:
			con.cursor().execute("DROP TABLE IF EXISTS [{}]".format(table))

			for df in pd.read_csv(file, chunksize=(1024 * 1024)):

				# Lowercase column names
				df.columns = map(str.strip, map(str.lower, df.columns))

				df = data_clean(df)
				df = data_in_polygon(df, manhattan)

				# Write to database
				df.to_sql(name=table, con=con, if_exists='append', index=False)


if __name__ == "__main__":
	download()
	tosqlite()

