
# RA, 2019-10-21

import os
import shutil
import functools
import requests
import numpy as np
from zipfile import ZipFile, ZIP_DEFLATED
from json import load as json_load


PARAM = {
	'api_url': "https://overpass-api.de/api/interpreter",
	# Note: check http://overpass-api.de/api/status

	'query': open("data/osm/manhattan/manhattan.ql", 'r').read(),

	'out_osm_archive': "data/manhattan/osm_json.zip",
	'zipped_name': "data",
}


def prepare_dirs():
	os.makedirs(os.path.dirname(PARAM['out_osm_archive']), exist_ok=True)


# def get_bbox():
# 	with connect(database) as con:
# 		bboxes = {
# 			table: [
# 				f(
# 					pd.read_sql(
# 						con=con,
# 						sql=(
# 							"select {fun}([{col}]) from [{table}]".format(
# 								fun=(f.__name__),
# 								col=col,
# 								table=table,
# 							)
# 						),
# 					).iloc[0, 0]
# 					for col in ["pickup_" + coor, "dropoff_" + coor]
# 				)
# 				for f in [min, max]
# 				for coor in ["latitude", "longitude"]
# 			]
# 			for (__, __, table) in src_trg_tbl()
# 		}
#
# 		json_dump(bboxes, open(bboxfile, 'w'))


def download_osm(to_file: str, overpass_query: str):
	if os.path.isfile(to_file):
		print("File {} already exists -- skipping download".format(to_file))
		return

	with requests.post(PARAM['api_url'], {'data': overpass_query}, stream=True) as response:
		if (response.status_code == 200):
			# https://github.com/psf/requests/issues/2155#issuecomment-50771010
			response.raw.read = functools.partial(response.raw.read, decode_content=True)
			with ZipFile(to_file, mode='w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
				with archive.open(PARAM['zipped_name'], mode='w', force_zip64=True) as fd:
					shutil.copyfileobj(response.raw, fd)
		else:
			print("Overpass status code: {} -- download aborted".format(response.status_code))


if __name__ == "__main__":
	prepare_dirs()
	download_osm(to_file=PARAM['out_osm_archive'], overpass_query=PARAM['query'])
