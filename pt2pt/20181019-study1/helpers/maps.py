
# RA, 2018-11-07

import io
import time
from helpers import commons

from PIL import Image

from math import pi, log, tan, exp, atan


# Convert geographical coordinates to pixels
# https://en.wikipedia.org/wiki/Web_Mercator_projection
# Note on google API:
# The world map is obtained with lat=lon=0, w=h=256, zoom=0
# Note on mapbox API:
# The world map is obtained with lat=lon=0, w=h=512, zoom=0

# Therefore:
ZOOM0_SIZE = 512 # Not 256

# Keep copies of downloaded maps
CACHEDIR = "../helpers/wget_cache/maps/"

# https://www.mapbox.com/api-documentation/#styles
mapbox_styles = {
	'streets-v10', 'outdoors-v10', 'light-v9', 'dark-v9', 'satellite-v9', 'satellite-streets-v10'
}

def g2p(lat, lon, zoom):
	return (
		# x
		ZOOM0_SIZE * (2 ** zoom) * (1 + lon / 180) / 2,
		# y
		ZOOM0_SIZE / (2 * pi) * (2 ** zoom) * (pi - log(tan(pi / 4 * (1 + lat / 90))))
	)

# Pixel to geo-coordinate
def p2g(x, y, zoom):
	return (
		# lat
		(atan(exp(pi - y / ZOOM0_SIZE * (2 * pi) / (2 ** zoom))) / pi * 4 - 1) * 90,
		# lon
		(x / ZOOM0_SIZE * 2 / (2 ** zoom) - 1) * 180,
	)

# bbox = (left, bottom, right, top)
def get_map_by_bbox(bbox, style='light-v9') :

	(left, bottom, right, top) = bbox

	assert(-180 <= left < right <= 180)
	assert(-90 <= bottom < top <= 90)

	(w, h) = (1024, 1024)
	(lat, lon) = ((top + bottom) / 2, (left + right) / 2)

	for zoom in range(16, 0, -1) :
		(x0, y0) = g2p(lat, lon, zoom)
		((TOP, LEFT), (BOTTOM, RIGHT)) = (p2g(x0 - w / 2, y0 - h / 2, zoom), p2g(x0 + w / 2, y0 + h / 2, zoom))
		if ((LEFT <= left < right <= RIGHT) and (BOTTOM <= bottom < top <= TOP)) :
			break

	token = open(".credentials/UV/mapbox-token.txt", 'r').read()

	retina = { True : "@2x", False : "" }[False]
	url = "https://api.mapbox.com/styles/v1/mapbox/{style}/static/{lon},{lat},{zoom}/{w}x{h}{retina}?access_token={token}&attribution=true&logo=false"
	url = url.format(style=style, lat=lat, lon=lon, token=token, zoom=zoom, w=w, h=h, retina=retina)

	b = commons.wget(url, cachedir=CACHEDIR).bytes

	I = Image.open(io.BytesIO(b))

	# If the "retina" @2x parameter is used, the image is twice the size of the requested dimensions
	(W, H) = I.size
	assert((W, H) in [(w, h), (2*w, 2*h)])

	i = I.crop((
		round(W * (left - LEFT) / (RIGHT - LEFT)),
		round(H * (bottom - BOTTOM) / (TOP - BOTTOM)),
		round(W * (right - LEFT) / (RIGHT - LEFT)),
        round(H * (top - BOTTOM) / (TOP - BOTTOM)),
	))

	return i

def mess() :
	# (lat, lon, zoom, w, h) = (22.6316, 120.358, 14, 500, 300)
	# Corners: (120.33654232788086, 22.64348272090081) (120.37945767211914, 22.61971625158942)


	# Kaohsiung (left, bottom, right, top)
	bbox = (120.2593, 22.5828, 120.3935, 22.6886)
	(left, bottom, right, top) = bbox

	import matplotlib.pyplot as plt
	plt.ion()
	plt.gca().axis([left, right, bottom, top])

	i = get_map_by_bbox(bbox)

	plt.gca().imshow(i, extent=(left, right, bottom, top), interpolation='quadric')
	plt.show()

	# i = Image.open("500x300.png")
	# plt.gca().axis([left, right, bottom, top])
	# f = plt.gcf()
	# ax = f.add_axes([0, 0.1, 0.7, 0.5])
	# #ax.plot([1, 2], [3, 4])
	# ax.imshow(i, interpolation='quadric')
	# ax.axis('off')
	# plt.show()


	fn = "../OUTPUT/13/Kaohsiung/UV/001-V3.json"
	import json
	J = json.load(open(fn, 'r'))
	for b in J :
		(Y, X) = (b['PositionLat'], b['PositionLon'])
		for (x, y) in zip(X, Y) :
			h = plt.plot(x, y, 'ro')
			plt.pause(0.1)
			time.sleep(0.1)
			h[0].remove()


if (__name__ == "__main__") :
	mess()
