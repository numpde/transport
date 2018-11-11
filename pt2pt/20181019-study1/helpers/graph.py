
# RA, 2018-11-11

import pickle
import geopy.distance
import sklearn.neighbors

# Metric for (lat, lon) coordinates
def geodesic(a, b) :
	return geopy.distance.geodesic(a, b).m

def foo() :

	osm_graph_file = "../OUTPUT/02/UV/kaohsiung.pkl"

	print("Loading OSM...")
	OSM = pickle.load(open(osm_graph_file, 'rb'))

	# Road network
	G = OSM['G']

	# Locations of the graph nodes
	node_pos = OSM['locs']

	print("Constructing the KNN tree...")
	(I, X) = (node_pos.keys(), node_pos.values())
	knn = {
		'ID-vec' : I,
		'tree' : sklearn.neighbors.BallTree(X, leaf_size=30, metric='pyfunc', func=geodesic)
	}

	# Nearest neighbors


if (__name__ == "__main__") :
	foo()
