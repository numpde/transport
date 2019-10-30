
# RA, 2019-10-29

import os
import inspect

from multiprocessing.pool import Pool


# Return caller's function name
def myname():
	return inspect.currentframe().f_back.f_code.co_name


# Create path leading to file
def makedirs(filename):
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	return filename


# Parallel computation map
def parallel_map(func, generator) -> list:
	with Pool(6) as pool:
		return list(pool.imap(func, generator, chunksize=100))
		# return list(pool.map(func, generator))

