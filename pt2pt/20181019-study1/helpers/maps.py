
# RA, 2018-11-07

import io
from enum import Enum
from helpers import commons

from PIL import Image

from math import pi, log, tan, exp, atan


# Convert geographical coordinates to pixels
# https://en.wikipedia.org/wiki/Web_Mercator_projection
# Note on google API:
# The world map is obtained with lat=lon=0, w=h=256, zoom=0
# Note on mapbox API:
# The world map is obtained with lat=lon=0, w=h=512, zoom=0
#
# Therefore:
MAPBOX_ZOOM0_SIZE = 512 # Not 256

# Keep copies of downloaded maps
CACHEDIR = "../helpers/wget_cache/maps/"

# https://www.mapbox.com/api-documentation/#styles
class MapBoxStyle(Enum) :
	streets = 'streets-v10'
	outdoors = 'outdoors-v10'
	light = 'light-v9'
	dark = 'dark-v9'
	satellite = 'satellite-v9'
	satellite_streets = 'satellite-streets-v10'

# Geo-coordinate in degrees => Pixel coordinate
def g2p(lat, lon, zoom):
	return (
		# x
		MAPBOX_ZOOM0_SIZE * (2 ** zoom) * (1 + lon / 180) / 2,
		# y
		MAPBOX_ZOOM0_SIZE / (2 * pi) * (2 ** zoom) * (pi - log(tan(pi / 4 * (1 + lat / 90))))
	)

# Pixel coordinate => geo-coordinate in degrees
def p2g(x, y, zoom):
	return (
		# lat
		(atan(exp(pi - y / MAPBOX_ZOOM0_SIZE * (2 * pi) / (2 ** zoom))) / pi * 4 - 1) * 90,
		# lon
		(x / MAPBOX_ZOOM0_SIZE * 2 / (2 ** zoom) - 1) * 180,
	)

# bbox = (left, bottom, right, top) in degrees
def get_map_by_bbox(bbox, token=None, style=MapBoxStyle.light) :

	if not token :
		raise RuntimeError("An API token is required")

	# Get the actual value from the enum class
	style = style.value

	# The region of interest in geo-coordinates in degrees
	(left, bottom, right, top) = bbox
	# Sanity check
	assert(-180 <= left < right <= 180)
	assert(-90 <= bottom < top <= 90)

	# The center point of the region of interest
	(lat, lon) = ((top + bottom) / 2, (left + right) / 2)

	# Rendered image map size in pixels as it should come from MapBox (no retina)
	(w, h) = (1024, 1024)

	# Look for appropriate zoom level to cover the region of interest by that map
	for zoom in range(16, 0, -1) :
		# Center point in pixel coordinates at this zoom level
		(x0, y0) = g2p(lat, lon, zoom)
		# The geo-region that the downloaded map would cover
		((TOP, LEFT), (BOTTOM, RIGHT)) = (p2g(x0 - w / 2, y0 - h / 2, zoom), p2g(x0 + w / 2, y0 + h / 2, zoom))
		# Would the map cover the region of interest?
		if (LEFT <= left < right <= RIGHT) and (BOTTOM <= bottom < top <= TOP) :
			break

	# Choose "retina" quality of the map
	retina = { True : "@2x", False : "" }[False]

	# Assemble the query URL
	url = "https://api.mapbox.com/styles/v1/mapbox/{style}/static/{lon},{lat},{zoom}/{w}x{h}{retina}?access_token={token}&attribution=false&logo=false"
	url = url.format(style=style, lat=lat, lon=lon, token=token, zoom=zoom, w=w, h=h, retina=retina)

	# Download the rendered image
	b = commons.wget(url, cachedir=CACHEDIR).bytes

	# Convert bytes to image object
	I = Image.open(io.BytesIO(b))

	# If the "retina" @2x parameter is used, the image is twice the size of the requested dimensions
	(W, H) = I.size
	assert((W, H) in [(w, h), (2*w, 2*h)])

	# Extract the map of the region of interest from the covering map
	i = I.crop((
		round(W * (left - LEFT) / (RIGHT - LEFT)),
		round(H * (bottom - BOTTOM) / (TOP - BOTTOM)),
		round(W * (right - LEFT) / (RIGHT - LEFT)),
        round(H * (top - BOTTOM) / (TOP - BOTTOM)),
	))

	return i
