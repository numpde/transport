#!/usr/bin/python3

# RA, 2018-10-21

## ================== IMPORTS :

import pickle
import inspect


## ==================== NOTES :

pass


## ==================== INPUT :

IFILE = {
	'OSM-pickled' : "OUTPUT/02/UV/kaohsiung.pkl",
}


## =================== OUTPUT :

OFILE = {
	'' : "",
}


## ==================== PARAM :

PARAM = {
	'' : 0,
}

## ====================== AUX :

# https://stackoverflow.com/questions/34491808/how-to-get-the-current-scripts-code-in-python
THIS = inspect.getsource(inspect.getmodule(inspect.currentframe()))

# Log which files are opened
def logged_open(filename, mode='r', *argv, **kwargs) :
	print("({}):\t{}".format(mode, filename))
	return open(filename, mode, *argv, **kwargs)


## ===================== WORK :

class BusstopTransit(object) :

	def __init__(self) :
		pass

	def __del__(self) :
		pass

	def init_graph_from(self, filename) :

		self.osm = pickle.load(logged_open(filename, 'rb'))


def test1() :
	bt = BusstopTransit()
	bt.init_graph_from(IFILE['OSM-pickled'])


## ==================== ENTRY :

if (__name__ == "__main__") :

	test1()

