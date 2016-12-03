#!/usr/bin/env python3

class navigator :
	name = "Greedy"

	def __init__(self, C, N) :
		assert isinstance(C, int) and (C >= 1)
		assert isinstance(N, int) and (N >= 1)
		# Capacity of the bus (integer >= 1)
		self.C = C
		# Number of stations (integer >= 1)
		self.N = N

	def step(self, b, B, Q) :
		# INPUT:
		# See check_input(..) for details.
		# The function may not modify b, B or Q.
		#
		# OUTPUT:
		# The function should return M, s
		# See check_output(..) for details.

		self.check_input(b, B, Q)

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

		self.check_output(b, B, Q, M, s)
		return M, s


	def is_station(self, n) :
		return isinstance(n, int) and ((0 <= n) and (n < self.N))

	def check_input(self, b, B, Q) :
		#
		# 1.
		# b is an integer 0 <= b < N denoting
		#   the current location of the bus.
		#
		assert self.is_station(b)
		#
		# 2.
		# B is a list [n1, n2, ..] of
		#   the destinations of the passengers
		#   currently on the bus
		#   (not exceeding the capacity), i.e.
		#   nk is the destination of passenger k.
		#   The order is that of boarding
		#   (provided by this function: see M).
		#   No destination is the current position.
		#
		assert isinstance(B, list)
		assert all(self.is_station(n) for n in B)
		assert all((n != b) for n in B)
		#
		# 3.
		# Q is a list of N lists, where
		#   Q[n] = [t1, t2, ..] is the list of
		#   people currently waiting at station n
		#   with destinations t1, t2, ..
		#   No destination equals the location,
		#   i.e. (t != n) for any t in Q[n]
		#
		assert isinstance(Q, list)
		assert (len(Q) == self.N)
		assert all(isinstance(q, list) for q in Q)
		assert all(all(self.is_station(t) for t in q) for q in Q)
		assert all(all((t != n) for t in q) for n, q in enumerate(Q))

	def check_output(self, b, B, Q, M, s) :
		#
		# 1.
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
		assert isinstance(M, list)
		assert all(isinstance(i, int) for i in M)
		assert all(((0 <= i) and (i < len(Q[b]))) for i in M)
		assert (len(B) + len(M) <= self.C)
		#
		# 2.
		# s is either +1, -1, or 0, indicating
		#   the direction of travel of the bus
		#   (the next station is (b + s) % N).
		#
		assert isinstance(s, int)
		assert (s in [-1, 0, 1])




