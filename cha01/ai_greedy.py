#!/usr/bin/env python3

# Date:   2016-12-04
# Author: verybusybus.wordpress.com

class navigator :
	name = "Greedy"

	def __init__(self, C, N) :
		self.C = C # Capacity of the bus
		self.N = N # Number of stations

	def step(self, b, B, Q) :

		# Number of passengers to board
		n = min(len(Q[b]), self.C - len(B))
		# Passenger selection from Q[b]
		M = list(range(n))

		# No passengers?
		if ((not B) and (not M)) : return [], 1

		# Next passenger's destination
		t = (B + [Q[b][i] for i in M])[0]

		# Destination relative to current position
		t = (self.N/2 - ((t-b) % self.N))

		# Move towards that destination
		s = (+1) if (t > 0) else (-1)

		return M, s


