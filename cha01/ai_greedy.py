#!/usr/bin/env python3

class navigator :
	name = "Greedy"
	
	def __init__(self, C, N) :
		# Capacity of the bus (integer >= 1)
		self.C = C
		# Number of stations (integer >= 1)
		self.N = N

	def step(self, b, B, Q) :
		# INPUT:
		#
		# b is an integer 0 <= b < N denoting
		#   the current location of the bus.
		#
		# B is a list [b1, b2, ..] of
		#   passengers currently on the bus
		#   (not exceeding the capacity), where
		#   bk is the destination of passenger k.
		#   The order is that of boarding
		#   (provided by this function: see M).
		#
		# Q is a list of N lists, where
		#   Q[n] = [t1, t2, ..] is the list of
		#   people currently waiting at station n
		#   with destinations t1, t2, ..
		#
		# OUTPUT:
		#
		# return M, s where
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



