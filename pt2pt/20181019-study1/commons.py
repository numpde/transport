
# RA, 2018-11-01

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np

# Find a way through matrix M bottom-to-top with right-to-left drift
# that minimizes the sum of entries (using dynamic programming)
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
	import numpy as np
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
						# go left
						(s(i, j - 1), i, j - 1),
						# go up
						(s(i - 1, j) + M[i, j], i - 1, j)
					)
				)

	# Retrace the optimal way
	match = [None] * M.shape[0]
	while (i >= 0) :
		M[i, j] = max(M.flatten()) # For visualization below
		match[i] = j
		(i, j) = (I[i, j], J[i, j])

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
	for f in OFILE.values() :
		os.makedirs(os.path.dirname(f), exist_ok=True)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Print which files are opened
def logged_open(filename, mode='r', *argv, **kwargs) :
	print("({}):\t{}".format(mode, filename))
	return open(filename, mode, *argv, **kwargs)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from collections import defaultdict

# Index an iterable _I_ of dict's by the return value of key_func
def index_dicts_by_key(I, key_func, collapse_repetitive=True, preserve_singletons=[]) :
	J = defaultdict(lambda: defaultdict(list))

	for i in I :
		for (k, v) in i.items() :
			J[key_func(i)][k].append(v)

	if collapse_repetitive :
		for (j, i) in J.items() :
			for (k, V) in i.items() :
				V = [json.loads(v) for v in set(json.dumps(v) for v in V)]
				if (1 == len(V)) :
					if k in preserve_singletons :
						J[j][k] = V
					else :
						J[j][k] = V.pop()

	# Convert all defaultdict to dict
	J = json.loads(json.dumps(J))

	return J

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Does the collection L contain mutually distinct elements?
def is_distinct(L) :
	return (len(set(L)) == len(list(L)))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import sys

def parse_options(OPTIONS) :

	if not OPTIONS :
		raise RuntimeError("No options to choose from")

	if (1 == len(sys.argv)) :

		if (1 == len(OPTIONS)) :
			(opt, fun) = OPTIONS.popitem()
			fun()
			return True
		else :
			raise RuntimeError("No option provided. Options are: {}".format(", ".join(OPTIONS.keys())))

	else :

		(opt, args) = (sys.argv[1], sys.argv[2:])

		if opt in OPTIONS :
			OPTIONS[opt](*args)
			return True


	raise RuntimeError("Unrecognized command line option")

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

def zipjson_load(fn) :

	assert(type(fn) is str), "This expects a file name"

	if not os.path.isfile(fn) :
		raise FileNotFoundError("File not found: {}".format(fn))

	if not os.stat(fn).st_size :
		# The file is empty. Choose not to return an empty JSON.
		raise EOFError("File {} is empty".format(fn))

	try :
		J = json.load(open(fn, 'r'))
	except :
		print("Exception while loading {}".format(fn))
		raise

	try :
		J = ZIPJSON(J).try_dec()
	except :
		print("Exception while decoding {}".format(fn))
		raise

	return J

def zipjson_dump(J, fn) :
	assert(type(fn) is str)
	E = ZIPJSON(J).enc()
	assert(json.dumps(E))
	return json.dump(E, open(fn, 'w'))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

