
# RA, 2018-11-07

import io
import urllib.request

from PIL import Image

from math import pi, log, tan, exp, atan


# Convert geographical coordinates to pixels
# https://en.wikipedia.org/wiki/Web_Mercator_projection
def g2p(lat, lon, zoom):
	return (
		# x
		256 / (2 * pi) * (2 ** zoom) * pi * (1 + lon / 180),
		# y
		256 / (2 * pi) * (2 ** zoom) * (pi - log(tan(pi / 4 * (1 + lat / 90))))
	)

# Pixel to geo-coordinate
def p2g(x, y, zoom):
	return (
		# lat
		(atan(exp(pi - y / 256 * (2 * pi) / (2 ** zoom))) / pi * 4 - 1) * 90,
		# lon
		(x / 256 * (2 * pi) / (2 ** zoom) / pi - 1) * 180,
	)

# bbox = (left, bottom, right, top)
def get_map_by_bbox(bbox) :

	(left, bottom, right, top) = bbox

	assert(left < right)
	assert(bottom < top)

	w = 600
	h = 500
	(lat, lon) = ((top + bottom) / 2, (left + right) / 2)

	for zoom in range(16, 0, -1) :
		(x0, y0) = g2p(lat, lon, zoom)
		((TOP, LEFT), (BOTTOM, RIGHT)) = (p2g(x0 - w / 2, y0 - h / 2, zoom), p2g(x0 + w / 2, y0 + h / 2, zoom))
		ZOOM = zoom - 1 # DO NOT KNOW WHY (zoom - 1) IS NECESSARY IN THE API CALL
		if ((LEFT <= left < right <= RIGHT) and (BOTTOM <= bottom < top <= TOP)) :
			del zoom
			break

	token = open("../.credentials/UV/mapbox-token.txt", 'r').read()


	url = "https://api.mapbox.com/styles/v1/mapbox/streets-v10/static/{lon},{lat},{zoom}/{w}x{h}?access_token={token}&attribution=true&logo=false".format(lat=lat, lon=lon, token=token, zoom=ZOOM, w=w, h=h)

	with urllib.request.urlopen(url) as response:
		b = response.read()

	i = Image.open(io.BytesIO(b))

	# print(LEFT, BOTTOM, RIGHT, TOP)
	# import matplotlib.pyplot as plt
	# plt.clf()
	# plt.imshow(i, extent=(LEFT, RIGHT, BOTTOM, TOP))
	# plt.show()
	# exit(39)

	assert((w, h) == i.size)

	i = i.crop((
		round(w * (left - LEFT) / (RIGHT - LEFT)),
		round(h * (bottom - BOTTOM) / (TOP - BOTTOM)),
		round(w * (right - LEFT) / (RIGHT - LEFT)),
        round(h * (top - BOTTOM) / (TOP - BOTTOM)),
	))

	return i

def mess() :
	# (lat, lon, zoom, w, h) = (22.6316, 120.358, 14, 500, 300)
	# Corners: (120.33654232788086, 22.64348272090081) (120.37945767211914, 22.61971625158942)

	# Kaohsiung (left, bottom, right, top)
	bbox = (120.2593, 22.5828, 120.3935, 22.6886)
	(left, bottom, right, top) = bbox

	import matplotlib.pyplot as plt
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


if (__name__ == "__main__") :
	mess()
