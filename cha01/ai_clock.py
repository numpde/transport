#!/usr/bin/env python3

# Date:   2016-12-04
# Author: verybusybus.wordpress.com

class navigator :
	name = "Clock"

	def __init__(self, C, N) :
		# Capacity of the bus (integer >= 1)
		self.C = C
		# Number of stations (integer >= 1)
		self.N = N

	def step(self, b, B, Q) :
		# INPUT
		#
		# b is an integer 0 <= b < N denoting
		#   the current location of the bus.
		#
		# B is a list [n1, n2, ..] of
		#   the destinations of the passengers
		#   currently on the bus
		#   (not exceeding the capacity), i.e.
		#   nk is the destination of passenger k.
		#   The order is that of boarding
		#   (provided by this function: see M).
		#   No destination is the current position.
		#
		# Q is a list of N lists, where
		#   Q[n] = [t1, t2, ..] is the list of
		#   people currently waiting at station n
		#   with destinations t1, t2, ..
		#   No destination equals the location,
		#   i.e. (t != n) for any t in Q[n].
		#
		# The input variable may be modified 
		# within this function w/o consequence.
		#
		#
		# OUTPUT
		#
		# The function should return M, s
		# where:
		#
		# M is a list of indices M = [i1, i2, .., im]
		#   into the list Q[b] indicating that
		#   the people Q[b][i] will board the bus
		#   (in the order defined by M).
		#   Set M = [] if no one boards the bus.
		#   Note the constraints:
		#     len(B) + len(M) <= Capacity C,
		#   and
		#     0 <= i < len(Q[b]) for each i in M.
		#
		# s is either +1, -1, or 0, indicating
		#   the direction of travel of the bus
		#   (the next station is (b + s) % N).
		
		
		# Number of passengers to board
		n = min(len(Q[b]), self.C - len(B))
		# Passenger selection from Q[b]:
		# Take passengers number 0, 1, .., n-1
		M = list(range(n))
		
		# Always go in one direction
		s = +1
		
		return M, s


