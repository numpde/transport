#!/usr/bin/python3

# RA, 2018-12-16

import datetime as dt
from enum import Enum

class Loc :
	# Expect:
	# 't' to be a timepoint
	# 'x' to be a (lat, lon) pair
	# 'desc' some descriptor of the location
	def __init__(self, t=None, x=None, desc=None) :
		t: dt.datetime
		if t and t.tzinfo :
			t = t.astimezone(dt.timezone.utc).replace(tzinfo=None)
		self.t = t
		self.x = x
		self.desc = desc
	def __str__(self) :
		#return "<Location '{}' at {}>".format(self.desc, self.x)
		return "{}/{} at {}".format(self.desc, self.x, self.t)
	def __repr__(self) :
		return "Loc(t={}, x={}, desc={})".format(self.t, self.x, self.desc)


class Mode(Enum) :
	walk = "Walk"
	bus = "Bus"


class Leg :
	def __init__(self, P: Loc, Q: Loc, mode: Mode, desc=None) :
		self.P = P
		self.Q = Q
		self.mode = mode
		self.desc = desc
	def __str__(self) :
		return "({P})--[{mode}/{desc}]-->({Q})".format(P=self.P, Q=self.Q, mode=self.mode, desc=self.desc)
