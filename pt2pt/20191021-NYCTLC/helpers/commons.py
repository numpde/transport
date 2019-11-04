
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
		# return list(pool.imap(func, generator, chunksize=100))
		return list(pool.map(func, generator))


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


class Range():
	def __call__(self, *args):
		return self.builtin_function(*args)
	def __getitem__(self, args):
		(a, b, s) = ([1, args, 1] if (type(args) is int) else [*args, 1] if (2 == len(args)) else args)
		return self.builtin_function(a, b + 1, s)
	def __init__(self, builtin_function):
		self.builtin_function = builtin_function
		def check_equal(a, b):
			(a, b) = (tuple(builtins.range(100)[x] if (type(x) is builtins.slice) else x) for x in [a, b])
			assert(a == b)
		check_equal(self(3), builtin_function(0, 3))
		check_equal(self[3], builtin_function(1, 3 + 1))
		check_equal(self(2, 6), builtin_function(2, 6))
		check_equal(self[2, 6], builtin_function(2, 6 + 1))
		check_equal(self(0, 6, 2), builtin_function(0, 6, 2))
		check_equal(self[0, 6, 2], builtin_function(0, 6 + 1, 2))

range = Range(builtins.range)
slice = Range(builtins.slice)
