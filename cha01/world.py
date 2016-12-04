#!/usr/bin/env python3

# Date:   2016-12-04
# Author: verybusybus.wordpress.com

from random import randint
from numpy import mean

class World :

	def __init__(self, C, N) :
		self.C = C         # Bus capacity
		self.N = N         # Number of stations
		self.b = None      # Bus position
		self.B = None      # Bus passengers' destinations [list]
		self.Q = None      # Queues at stations [list of list]
		self.i = None      # Iteration number (time)
		self.NEWS = [None] # World trajectory record [list of tuple/None]
		self.rewind()

	def rewind(self) :
		self.b = 0
		self.B = []
		self.Q = [[] for _ in range(self.N)]
		self.i = 0

	def news(self) :
		# Create news if necessary
		if (len(self.NEWS) <= self.i) :
			# New person arrives at "a" with destination "b"
			a = randint(0, self.N-1)
			b = (a + randint(1, self.N-1)) % self.N
			self.NEWS.append((a, b))
		assert((0 <= self.i) and (self.i < len(self.NEWS)))
		return self.NEWS[self.i]

	def look(self) :
		# Return a copy of (b, B, Q)
		return self.b, self.B[:], [q[:] for q in self.Q]

	def move(self, M, s) :
		# Check consistency from time to time
		if (randint(0, 100) == 0) :
			self.check_consistency(self.C, self.N, self.b, self.B, self.Q, M, s)

		# Passengers mount (in the given order)
		for m in M : self.B.append(self.Q[self.b][m])
		# Remove them from the queue
		for m in sorted(M, reverse=True) : self.Q[self.b].pop(m)

		# Advance bus
		self.b = (self.b + s) % self.N

		# Passengers unmount
		self.B = [p for p in self.B if (p != self.b)]

		# Advance time
		self.i += 1

		assert(self.news() is not None)
		# New person arrives at "a" with destination "b"
		a, b = self.news()
		# Queue in the new person
		self.Q[a].append(b)

	def get_w(self) :
		# Number of people waiting in queue, averaged over the stations
		return mean([len(P) for P in self.Q])

	@staticmethod
	def check_consistency(C, N, b, B, Q, M, s) :
		
		# 0.
		# C is an integer >= 1
		# N is an integer >= 1
		
		assert isinstance(C, int) and (C >= 1)
		assert isinstance(N, int) and (N >= 1)
		
		is_station = lambda n : isinstance(n, int) and ((0 <= n) and (n < N))

		# 1.
		# b is an integer 0 <= b < N denoting
		#   the current location of the bus.

		assert is_station(b)

		# 2.
		# B is a list [n1, n2, ..] of
		#   the destinations of the passengers
		#   currently on the bus
		#   (not exceeding the capacity), i.e.
		#   nk is the destination of passenger k.
		#   The order is that of boarding
		#   (provided by this function: see M).
		#   No destination is the current position.

		assert isinstance(B, list)
		assert all(is_station(n) for n in B)
		assert all((n != b) for n in B)

		# 3.
		# Q is a list of N lists, where
		#   Q[n] = [t1, t2, ..] is the list of
		#   people currently waiting at station n
		#   with destinations t1, t2, ..
		#   No destination equals the location,
		#   i.e. (t != n) for any t in Q[n].

		assert isinstance(Q, list)
		assert (len(Q) == N)
		assert all(isinstance(q, list) for q in Q)
		assert all(all(is_station(t) for t in q) for q in Q)
		assert all(all((t != n) for t in q) for n, q in enumerate(Q))

		# 4.
		# M is a list of indices M = [i1, i2, .., im]
		#   into the list Q[b] indicating that
		#   the people Q[b][i] will board the bus
		#   (in the order defined by M).
		#   Set M = [] if no one boards the bus.
		#   Note the constraints:
		#     len(B) + len(M) <= Capacity C,
		#   and
		#     0 <= i < len(Q[b]) for each i in M.

		assert isinstance(M, list)
		assert all(isinstance(i, int) for i in M)
		assert all(((0 <= i) and (i < len(Q[b]))) for i in M)
		assert (len(B) + len(M) <= C)

		# 5.
		# s is either +1, -1, or 0, indicating
		#   the direction of travel of the bus
		#   (the next station is (b + s) % N).
		#
		assert isinstance(s, int)
		assert (s in [-1, 0, 1])



