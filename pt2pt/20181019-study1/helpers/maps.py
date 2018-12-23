
# RA, 2018-11-07

import io

from enum import Enum
from PIL import Image
from math import pi, log, tan, exp, atan, log2, floor
from urllib.error import URLError

import matplotlib as mpl

from helpers import commons

PARAM = {
	'do_retina' : True,
	'do_snap_to_dyadic' : False,
}

# Convert geographical coordinates to pixels
# https://en.wikipedia.org/wiki/Web_Mercator_projection
# Note on google API:
# The world map is obtained with lat=lon=0, w=h=256, zoom=0
# Note on mapbox API:
# The world map is obtained with lat=lon=0, w=h=512, zoom=0
#
# Therefore:
MAPBOX_ZOOM0_SIZE = 512 # Not 256

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

def ax2mb(left, right, bottom, top) :
	return (left, bottom, right, top)

def mb2ax(left, bottom, right, top) :
	return (left, right, bottom, top)

def bbox_for_points(pp) :
	(left, bottom, right, top) = [
		min(lon for (lat, lon) in pp),
		min(lat for (lat, lon) in pp),
		max(lon for (lat, lon) in pp),
		max(lat for (lat, lon) in pp)
	]
	return (left, bottom, right, top)

# bbox = (left, bottom, right, top) in degrees
def get_map_by_bbox(bbox, token=None, style=MapBoxStyle.light, cachedir=None) :

	if not token :
		raise RuntimeError("An API token is required")

	# The region of interest in geo-coordinates in degrees
	(left, bottom, right, top) = bbox
	# Sanity check
	assert(-90 <= bottom < top <= 90)
	assert(-180 <= left < right <= 180)

	# Rendered image map size in pixels as it should come from MapBox (no retina)
	(w, h) = (1024, 1024)

	# The center point of the region of interest
	(lat, lon) = ((top + bottom) / 2, (left + right) / 2)

	# Reduce precision of (lat, lon) to increase cache hits
	if PARAM['do_snap_to_dyadic'] :
		snap_to_dyadic = (lambda a, b : (lambda x, scale=(2 ** floor(log2(abs(b - a) / 4))) : (round(x / scale) * scale)))
		lat = snap_to_dyadic(bottom, top)(lat)
		lon = snap_to_dyadic(left, right)(lon)

		assert((bottom < lat < top) and (left < lon < right)), "Reference point not inside the region of interest"

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
	retina = { True : "@2x", False : "" }[PARAM['do_retina']]

	# Assemble the query URL
	url = "https://api.mapbox.com/styles/v1/mapbox/{style}/static/{lon},{lat},{zoom}/{w}x{h}{retina}?access_token={token}&attribution=false&logo=false"
	url = url.format(style=style.value, lat=lat, lon=lon, token=token, zoom=zoom, w=w, h=h, retina=retina)

	# Download the rendered image
	b = commons.wget(url, cachedir=cachedir).bytes

	# Convert bytes to image object
	I = Image.open(io.BytesIO(b), mode='r')

	# # DEBUG: show image
	# import matplotlib as mpl
	# mpl.use("TkAgg")
	# import matplotlib.pyplot as plt
	# plt.imshow(I)
	# plt.show()
	# exit(39)

	# If the "retina" @2x parameter is used, the image is twice the size of the requested dimensions
	(W, H) = I.size
	assert((W, H) in [(w, h), (2*w, 2*h)])

	# Extract the map of the region of interest from the covering map
	i = I.crop((
		round(W * (left - LEFT) / (RIGHT - LEFT)),
        round(H * (top - TOP) / (BOTTOM - TOP)),
		round(W * (right - LEFT) / (RIGHT - LEFT)),
		round(H * (bottom - TOP) / (BOTTOM - TOP)),
	))

	return i


def write_track_img(waypoints, tracks, fd, mapbox_api_token=None, plotter=None, dpi=180) :
	mpl.use('Agg')
	import matplotlib.pyplot as plt

	ax : plt.Axes
	fig : plt.Figure
	(fig, ax) = plt.subplots()

	ax.tick_params(axis='both', which='both', labelsize='xx-small')

	def default_plotter(fig, ax) :
		if (len(tracks) == 1) :
			(y, x) = zip(*tracks[0])
			ax.plot(x, y, 'b-', linewidth=2)
			ax.plot(x[0], y[0], 'o', c='g', markersize=3)
			ax.plot(x[-1], y[-1], 'o', c='r', markersize=3)
		else :
			for track in tracks :
				(y, x) = zip(*track)
				ax.plot(x, y, '-', linewidth=1, alpha=0.3, markersize=1.5)
			for track in tracks :
				(y, x) = zip(*track)
				ax.plot(x[0], y[0], 'o', c='g', markersize=1)
				ax.plot(x[-1], y[-1], 'o', c='r', markersize=1)

		if waypoints :
			for (y, x) in waypoints :
				ax.plot(x, y, 'o', alpha=0.3, c='k', markersize=0.3)

	(plotter or default_plotter)(fig, ax)

	axis = commons.niceaxis(ax.axis(), expand=1.1)
	[i.set_fontsize(8) for i in ax.get_xticklabels() + ax.get_yticklabels()]
	ax.axis(axis)

	try :
		if mapbox_api_token :
			ax.imshow(get_map_by_bbox(ax2mb(*axis), token=mapbox_api_token), extent=axis, interpolation='quadric', zorder=-100)
	except URLError as e:
		print("Warning: no background map (no internet connection?)")

	fig.savefig(fd, dpi=dpi, bbox_inches='tight', pad_inches=0)
	plt.close(fig)
