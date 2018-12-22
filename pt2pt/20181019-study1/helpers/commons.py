
# RA, 2018-11-01

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from itertools import groupby

def sort_and_group(C, key=None) :
	return groupby(sorted(C, key=key), key=key)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# https://docs.python-guide.org/writing/logging/
# https://docs.python.org/3/library/logging.html
# https://stackoverflow.com/questions/3220284/how-to-customize-the-time-format-for-python-logging

#LoggingLevels = dict(CRITICAL=50, ERROR=40, WARNING=30, INFO=20, DEBUG=10, NOTSET=0)

import sys
import logging

def initialize_logger() :
	from logging.config import dictConfig

	for mod in ['PIL', 'matplotlib'] :
		logging.getLogger(mod).setLevel(logging.WARNING)

	dictConfig(dict(
		version = 1,
		formatters = {
			'f': {
				'format': "%(levelname)-8s [%(asctime)s] : %(message)s",
				'datefmt': "%Y%m%d %H:%M:%S %Z",
			},
		},
		handlers = {
			'h': {
				'class': "logging.StreamHandler",
				'formatter': "f",
				'level': logging.DEBUG,
				'stream': "ext://sys.stdout",
			},
		},
		root = {
			'handlers': ['h'],
			'level': logging.DEBUG,
		}
	))

	logger = logging.getLogger()

	return logger

logger = initialize_logger()

def test_logger() :
	print("Testing logger: Before")
	for f in [logger.debug, logger.info, logger.warning, logger.error, logger.critical] :
		f("OK")
	print("Testing logger: After")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class UponDel :
	def __init__(self, action) :
		self.action = action
	def __del__(self) :
		self.action()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import re

# Reverse a str.format operation, for example:
# from template="abc {x} def {y}" and instance="abc X def Y"
# make a dictionary {'x': X, 'y': Y}
def unformat(template: str, instance: str) :
	from string import Formatter
	K = [ bit[1] for bit in Formatter().parse(template) ]
	K = [ k for k in K if k ]
	template = template.replace(".", "\.")
	return dict(zip(K, re.fullmatch(template.format(**{k: "(.*)" for k in K}), instance).groups()))

# Undo a str.format operation, then redo it using the dict 'replace'
def reformat(template: str, instance: str, replace={}) :
	return template.format(**{**unformat(template, instance), **replace})

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def token_for(service: str) :
	service = service.lower()

	if ("mapbox" in service) :
		return open(".credentials/UV/mapbox-token.txt", 'r').readline().strip()

	raise ValueError("No token for service '{}'".format(service))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def is_truthy(x) :
	return bool(x)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# https://stackoverflow.com/a/50983362
def identity(x, *args) :
	return (x,) + args if args else x

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import json

# Format JSON structure nicely
def pretty_json(J) :
	return json.dumps(J, indent=2, ensure_ascii=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import glob

# List files
def ls(pattern) :
	return sorted(list(glob.glob(pattern, recursive=True)))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np

# Random subset of a list (without replacement by default)
def random_subset(a, weights=None, k=None, replace=False) :

	# Note:
	# Use indices b/c numpy.random.choice yields "ValueError: a must be 1-dimensional" for a list of tuples
	# It also expects the probabilities/weights to sum to one

	a = list(a)

	if weights :
		if sum(weights) :
			weights = [w / sum(weights) for w in weights]
		else :
			weights = None

	return list(a[i] for i in np.random.choice(len(a), size=k, p=weights, replace=replace))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import math

def niceaxis(axis, expand=1.1, minaspect=((1 + math.sqrt(5)) / 2)) :
	(left, right, bottom, top) = axis

	# Expand by some factor
	(left, right) = ((left + right) / 2 + s * ((right - left) / 2 * expand) for s in (-1, +1))
	(bottom, top) = ((bottom + top) / 2 + s * ((top - bottom) / 2 * expand) for s in (-1, +1))

	# Compute a nicer aspect ratio if it is too narrow
	(w, h) = (right - left, top - bottom)
	if (w < h / minaspect) : (left, right) = (((left + right) / 2 + s * h / minaspect / 2) for s in (-1, +1))
	if (h < w / minaspect) : (bottom, top) = (((bottom + top) / 2 + s * w / minaspect / 2) for s in (-1, +1))

	return (left, right, bottom, top)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from itertools import groupby

# Remove consecutive repeats
def remove_repeats(xx, key=None):
	# https://stackoverflow.com/a/5738933
	return [next(iter(g)) for (k, g) in groupby(xx, key)]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import random
import numpy as np

def seed(a=123) :
	random.seed(a)
	np.random.seed(a)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import time
import hashlib
import urllib.parse, urllib.request

# Class to fetch files from WWW
class wget :

	number_of_calls = 0
	THROTTLE_MAX_CALLS = 1000 # Max number of wget requests per session
	THROTTLE_INBETWEEN = 1 # Throttle time in seconds

	def __init__(self, url: str, cachedir=None) :

		assert(url), "Illegal URL parameter"

		# Encode potential Chinese characters
		url = urllib.parse.quote(url, safe=':/?&=,@')

		if cachedir :
			os.makedirs(cachedir, exist_ok=True)

			# Compress cache filename
			# https://stackoverflow.com/a/295150
			filename = cachedir + "/" + hashlib.sha256(url.encode('utf-8')).hexdigest()
		else :
			filename = None

		if filename :
			if os.path.isfile(filename) :
				# Cached result found
				# logger.debug("wget cachefile found ({})".format(filename))
				with open(filename, 'rb') as fd :
					self.bytes = fd.read()
				# logger.debug("wget cachefile read ({})".format(len(self.bytes)))
				return

		wget.number_of_calls = wget.number_of_calls + 1

		if (wget.number_of_calls > self.THROTTLE_MAX_CALLS) :
			raise RuntimeError("Call limit exceeded for wget")

		time.sleep(self.THROTTLE_INBETWEEN)

		with urllib.request.urlopen(url) as response :

			self.bytes = response.read()

			if filename :
				try :
					with open(filename, 'wb') as fd :
						fd.write(self.bytes)
				except IOError as e :
					print("Warning: error writing cache in wget ({})".format(e))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Use as follows:
# inspect( {'Data': ('Science', 'Rules')} )( {'Data': {'Science': True, 'Rules': False}} )
class inspect :

	def __init__(self, template) :
		self.keys = template

	def __extract(self, x, keys):
		if type(keys) is dict :
			assert(1 == len(keys)), "Only one parent key allowed"
			(k, subkeys) = next(iter(keys.items()))
			return self.__extract(x[k], subkeys)
		if type(keys) is tuple :
			return tuple(self.__extract(x, k) for k in keys)
		if type(keys) is list :
			return list(self.__extract(x, k) for k in keys)
		return x[keys]

	def __call__(self, x) :
		return self.__extract(x, self.keys)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np

# Find a way through matrix M bottom-to-top with right-to-left drift
# that minimizes the sum of entries (using dynamic programming).
# Returns a list 'match' such that row i is matched with column match[i].
#
# Recursion template:
#
# def sum(i, j) :
# 	if (i < 0) or (j < 0) : return 0
# 	return min(sum(i, j - 1), M[i, j] + sum(i - 1, j))
#
def align(M) :
	# Sum matrix
	S = 0 * M

	# These will record the trajectory
	I = np.zeros(M.shape, dtype=int)
	J = np.zeros(M.shape, dtype=int)

	def s(i, j) :
		if (i < 0) or (j < 0) : return 0
		return S[i, j]

	# Dynamic programing loops
	for i in range(0, M.shape[0]) :
		for j in range(0, M.shape[1]) :
			(S[i, j], I[i, j], J[i, j]) = \
				(
					# In the first column, can only go up
					(j == 0) and (s(i - 1, j) + M[i, j], i - 1, j)
				) or (
					# Otherwise have a choice:
					min(
						# go left, for free
						(s(i, j - 1), i, j - 1),
						# go up, at a cost of picking up M[i, j]
						(s(i - 1, j) + M[i, j], i - 1, j)
					)
				)

	# Retrace the optimal way
	match = [None] * M.shape[0]
	while (i >= 0) :
		match[i] = j
		(i, j) = (I[i, j], J[i, j])

	# For visualization
	for (i, j) in enumerate(match) :
		M[i, j] = -max(M.flatten()) / 5

	# # For visualization:
	# import matplotlib.pyplot as plt
	# plt.imshow(M)
	# plt.show()

	# Now: row i is matched with column match[i]
	return match

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import geopy.distance

# Metric for (lat, lon) coordinates
def geodesic(a, b) :
	return geopy.distance.geodesic(a, b).m

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os

# Create output directories
def makedirs(OFILE) :
	if type(OFILE) is str :
		try :
			os.makedirs(os.path.dirname(OFILE).format(), exist_ok=True)
			return OFILE
		except (IndexError, KeyError) as e :
			#print("makedirs failed ({})".format(e))
			return False

	if type(OFILE) is dict :
		return makedirs(OFILE.values())

	return all(makedirs(f) for f in OFILE)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Print which files are opened
def logged_open(filename, mode='r', *argv, **kwargs) :
	print("({}):\t{}".format(mode, filename))
	return open(filename, mode, *argv, **kwargs)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from collections import defaultdict

# Index an iterable _I_ of dict's by the return value of key_func
def index_dicts_by_key(I, key_func, collapse_repetitive=True, keys_dont_collapse=[], keys_singletons_ok=[]):
	J = defaultdict(lambda: defaultdict(list))

	for i in I :
		for (k, v) in i.items() :
			J[key_func(i)][k].append(v)

	if collapse_repetitive :
		for (j, i) in J.items() :
			for (k, V) in i.items() :
				if k in keys_dont_collapse : continue
				V = [json.loads(v) for v in set(json.dumps(v) for v in V)]
				if (1 == len(V)) :
					if k in keys_singletons_ok :
						J[j][k] = V
					else :
						J[j][k] = V.pop()

	# Convert all defaultdict to dict
	J = { k : dict(j) for (k, j) in J.items() }

	return J

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Does the collection L contain mutually distinct elements?
def all_distinct(L) :
	L = list(L)
	return (len(L) == len(set(L)))

# Does the collection L contain only copies of one element?
def all_samesame(L) :
	return (1 == len(set(L)))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os, json, zlib, base64

class ZIPJSON :

	TAGS = ['base64(zip(o))', 'base64zip']

	def __init__(self, data) :

		try :
			json.dumps(data)
		except :
			raise RuntimeError("The passed object not recognized as JSON")

		self.data = data

	def enc(self):

		if not self.data : return self.data

		if (type(self.data) is dict) :
			if (1 == len(self.data)) :
				if set.intersection(set(self.data.keys()), set(self.TAGS)) :
					print(self.data)
					raise RuntimeError("It appears that the input is compressed already")

		C = {
			self.TAGS[0] : base64.b64encode(
				zlib.compress(
					json.dumps(self.data).encode('utf-8')
				)
			).decode('ascii')
		}

		return C

	def try_dec(self) :

		if not (type(self.data) is dict) : return self.data

		def dec(D) :
			for (k, v) in D.items() :
				if k in self.TAGS :
					v = json.loads( zlib.decompress( base64.b64decode( v ) ) )
				yield (k, v)

		J = dict(dec(self.data))

		if (1 == len(J)) :
			if set.intersection(set(J.keys()), set(self.TAGS)) :
				(_, J) = J.popitem()

		return J

def zipjson_load(fn, opener=open) :

	assert(type(fn) is str), "This expects a file name"

	if not os.path.isfile(fn) :
		raise FileNotFoundError("File not found: {}".format(fn))

	if not os.stat(fn).st_size :
		# The file is empty. Choose not to return an empty JSON.
		raise EOFError("File {} is empty".format(fn))

	try :
		J = json.loads(opener(fn, 'rb').read())
	except :
		#print("Exception while loading {}".format(fn))
		raise

	try :
		J = ZIPJSON(J).try_dec()
	except :
		print("Exception while decoding {}".format(fn))
		raise

	return J

def zipjson_dump(J, fn, opener=open) :
	assert(type(fn) is str)
	E = ZIPJSON(J).enc()
	assert(json.dumps(E))
	return json.dump(E, opener(fn, 'w'))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import sys

def parse_options(OPTIONS, commands=sys.argv[1:]) :

	if not OPTIONS :
		raise ValueError("No options to choose from")

	if not commands :

		if (1 == len(OPTIONS)) :

			# No option provided, but there is only one to choose from
			(next(iter(OPTIONS.values())))()
			return True

	else :

		(opt, args) = (commands[0], commands[1:])

		if opt in OPTIONS :
			(OPTIONS[opt])(*args)
			return True

	print("Invalid or no option provided. Options are: {}".format(", ".join(OPTIONS.keys())))
	commands = [c for c in input("Select option: ").strip().split(' ') if c]

	return (commands and parse_options(OPTIONS, commands))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

