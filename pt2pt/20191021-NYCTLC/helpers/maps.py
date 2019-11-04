
# RA, 2018-11-07
# RA, 2019-10-24

# https://medium.com/@busybus/rendered-maps-with-python-ffba4b34101c

import io
import os

import urllib
WGET_TIMEOUT = 20  # In seconds

from enum import Enum
from PIL import Image
from math import pi, log, tan, exp, atan, log2, floor

from urllib.error import URLError
from socket import timeout as TimeoutError

from retry import retry

import percache
cache = percache.Cache("/tmp/percache_mapbox_maps", livesync=True)

# Load the MapBox token, if present
import dotenv
dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import matplotlib as mpl



PARAM = {
	'do_retina': True,
	'do_snap_to_dyadic': True,
}


# Convert geographical coordinates to pixels
# https://en.wikipedia.org/wiki/Web_Mercator_projection
#
# Note on google API:
# The world map is obtained with lat=lon=0, w=h=256, zoom=0
# Note on mapbox API:
# The world map is obtained with lat=lon=0, w=h=512, zoom=0
#
# Therefore:
MAPBOX_ZOOM0_SIZE = 512  # Not 256


# https://www.mapbox.com/api-documentation/#styles
class MapBoxStyle(Enum):
	streets = 'streets-v11'
	outdoors = 'outdoors-v11'
	light = 'light-v10'
	dark = 'dark-v10'
	satellite = 'satellite-v9'
	satellite_streets = 'satellite-streets-v11'
	navi_preview_day = 'navigation-preview-day-v4'
	navi_preview_night = 'navigation-preview-night-v4'
	navi_guide_day = 'navigation-guidance-day-v4'
	navi_guide_night = 'navigation-guidance-night-v4'


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


def ax2mb(left, right, bottom, top):
	return (left, bottom, right, top)


def mb2ax(left, bottom, right, top):
	return (left, right, bottom, top)


def bbox_for_points(pp):
	(left, bottom, right, top) = [
		min(lon for (lat, lon) in pp),
		min(lat for (lat, lon) in pp),
		max(lon for (lat, lon) in pp),
		max(lat for (lat, lon) in pp)
	]
	return (left, bottom, right, top)


@cache
@retry((URLError, TimeoutError), tries=3, delay=1)
def wget(url: str) -> bytes:
	with urllib.request.urlopen(url, timeout=WGET_TIMEOUT) as response:
		return response.read()


# bbox = (left, bottom, right, top) in degrees
def get_map_by_bbox(bbox, token=os.getenv("MAPBOX_TOKEN"), style=MapBoxStyle.light):

	if not token:
		raise RuntimeError("An API token is required")

	# The region of interest in geo-coordinates in degrees
	(left, bottom, right, top) = bbox
	# Sanity check
	assert (-90 <= bottom < top <= 90)
	assert (-180 <= left < right <= 180)

	# Rendered image map size in pixels as it should come from MapBox (no retina)
	(w, h) = (1024, 1024)

	# The center point of the region of interest
	(lat, lon) = ((top + bottom) / 2, (left + right) / 2)

	# Reduce precision of (lat, lon) to increase cache hits
	if PARAM['do_snap_to_dyadic']:
		snap_to_dyadic = (lambda a, b: (lambda x, scale=(2 ** floor(log2(abs(b - a) / 4))): (round(x / scale) * scale)))
		lat = snap_to_dyadic(bottom, top)(lat)
		lon = snap_to_dyadic(left, right)(lon)

		assert ((bottom < lat < top) and (left < lon < right)), "Reference point not inside the region of interest"

	# Look for appropriate zoom level to cover the region of interest by that map
	for zoom in range(16, 0, -1):
		# Center point in pixel coordinates at this zoom level
		(x0, y0) = g2p(lat, lon, zoom)
		# The geo-region that the downloaded map would cover
		((TOP, LEFT), (BOTTOM, RIGHT)) = (p2g(x0 - w / 2, y0 - h / 2, zoom), p2g(x0 + w / 2, y0 + h / 2, zoom))
		# Would the map cover the region of interest?
		if (LEFT <= left < right <= RIGHT) and (BOTTOM <= bottom < top <= TOP):
			break

	# Choose "retina" quality of the map
	retina = {True: "@2x", False: ""}[PARAM['do_retina']]

	# Assemble the query URL
	url = F"https://api.mapbox.com/styles/v1/mapbox/{style.value}/static/{lon},{lat},{zoom}/{w}x{h}{retina}?access_token={token}&attribution=false&logo=false"

	# Download the rendered image
	b = wget(url)

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
	assert ((W, H) in [(w, h), (2 * w, 2 * h)])

	# Extract the region of interest from the larger covering map
	i = I.crop((
		round(W * (left - LEFT) / (RIGHT - LEFT)),
		round(H * (top - TOP) / (BOTTOM - TOP)),
		round(W * (right - LEFT) / (RIGHT - LEFT)),
		round(H * (bottom - TOP) / (BOTTOM - TOP)),
	))

	return i


# def write_track_img(waypoints, tracks, fd, mapbox_api_token=None, plotter=None, dpi=180):
# 	mpl.use('Agg')
# 	import matplotlib.pyplot as plt
#
# 	ax: plt.Axes
# 	fig: plt.Figure
# 	(fig, ax) = plt.subplots()
#
# 	def default_plotter(fig, ax):
# 		if tracks:
# 			if (len(tracks) == 1):
# 				(y, x) = zip(*tracks[0])
# 				ax.plot(x, y, 'b-', linewidth=2)
# 				ax.plot(x[0], y[0], 'o', c='g', markersize=3)
# 				ax.plot(x[-1], y[-1], 'o', c='r', markersize=3)
# 			else:
# 				# Plot each track as a line
# 				for track in tracks:
# 					if not track: continue
# 					(y, x) = zip(*track)
# 					ax.plot(x, y, '-', linewidth=1, alpha=0.3, markersize=1.5)
# 				# Plot start and end points
# 				for track in tracks:
# 					if not track: continue
# 					(y, x) = zip(*track)
# 					ax.plot(x[0], y[0], 'o', c='g', markersize=1)
# 					ax.plot(x[-1], y[-1], 'o', c='r', markersize=1)
#
# 		if waypoints:
# 			for (y, x) in waypoints:
# 				ax.plot(x, y, 'o', alpha=0.5, c='k', markersize=0.7, zorder=10)
#
# 	(plotter or default_plotter)(fig, ax)
#
# 	# # https://stackoverflow.com/questions/14711655/how-to-prevent-numbers-being-changed-to-exponential-form-in-python-matplotlib-fi
# 	# ax.get_xaxis().get_major_formatter().set_scientific(False)
# 	# ax.get_yaxis().get_major_formatter().set_scientific(False)
# 	# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.ticklabel_format.html
# 	ax.ticklabel_format(axis='both', style='plain', useOffset=False)
#
# 	# Set small font size for tick labels
# 	ax.tick_params(axis='both', which='both', labelsize='xx-small')
# 	# [i.set_fontsize(5) for i in ax.get_xticklabels() + ax.get_yticklabels()]
#
# 	axis = commons.niceaxis(ax.axis(), expand=1.1)
# 	ax.axis(axis)
#
# 	try:
# 		if mapbox_api_token:
# 			ax.imshow(get_map_by_bbox(ax2mb(*axis), token=mapbox_api_token), extent=axis, interpolation='quadric',
# 					  zorder=-100)
# 	except (URLError, TimeoutError):
# 		commons.logger.warning("No background map (no internet connection?)")
#
# 	fig.savefig(fd, dpi=dpi, bbox_inches='tight', pad_inches=0)
# 	plt.close(fig)
