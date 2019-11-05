
# RA, 2019-10-29

import os
import inspect

import logging

from time import time as tic

from multiprocessing.pool import Pool

from contextlib import contextmanager
from typing import Tuple, ContextManager
import matplotlib.pyplot as plt

import builtins


# Return caller's function name
def myname():
	return inspect.currentframe().f_back.f_code.co_name


# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
# https://docs.python.org/3/library/inspect.html
def this_module_body(goback=1):
	return inspect.getsource(inspect.getmodule(inspect.stack()[goback].frame))


# Create path leading to file
def makedirs(filename) -> str:
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	return filename


# Parallel computation map
def parallel_map(func, generator) -> list:
	with Pool(6) as pool:
		return list(pool.imap(func, generator, chunksize=100))
		# return list(pool.map(func, generator))


# Context manager for plt.subplots(1, 1)
@contextmanager
def figax() -> ContextManager[Tuple[plt.Figure, plt.Axes]]:
	(fig, ax1) = plt.subplots()
	yield (fig, ax1)
	plt.close(fig)


@contextmanager
def section(description: str, print=None):
	fname = inspect.currentframe().f_back.f_back.f_code.co_name
	print and print(F"{fname} -- {description}")
	start = tic()
	yield
	print and print(F"{fname} -- {description} [{(tic() - start):.2g}s]")

