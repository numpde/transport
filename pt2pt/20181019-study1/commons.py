
# RA, 2018-11-01

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
def index_dicts_by_key(I, key_func, collapse_repetitive=True) :
	J = defaultdict(lambda: defaultdict(list))

	for i in I :
		for (k, v) in i.items() :
			J[key_func(i)][k].append(v)

	if collapse_repetitive :
		for (j, i) in J.items() :
			for (k, V) in i.items() :
				if (1 == len(set(json.dumps(v) for v in V))) :
					J[j][k] = next(iter(V))

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

	if (len(sys.argv) > 1) :

		(opt, args) = (sys.argv[1], sys.argv[2:])

		if opt in OPTIONS :
			OPTIONS[opt](*args)
			return True

	elif (1 == len(OPTIONS)) and (1 == len(sys.argv)) :

		(opt, fun) = OPTIONS.popitem()
		fun()
		return True

	raise RuntimeError("Unrecognized command line option")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import json, zlib, base64

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
	assert(type(fn) is str)
	J = json.load(open(fn, 'r'))
	J = ZIPJSON(J).try_dec()
	return J

def zipjson_dump(J, fn) :
	assert(type(fn) is str)
	E = ZIPJSON(J).enc()
	assert(json.dumps(E))
	return json.dump(E, open(fn, 'w'))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

