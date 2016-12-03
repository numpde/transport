#!/usr/bin/env python3

# R. Andreev, 2016-12-03

import random
from numpy import mean
#
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
			a = random.randint(0, self.N-1)
			b = (a + random.randint(1, self.N-1)) % self.N
			self.NEWS.append((a, b))
		assert((0 <= self.i) and (self.i < len(self.NEWS)))
		return self.NEWS[self.i]

	def look(self) :
		# Return a copy of (b, B, Q)
		return self.b, self.B[:], [q[:] for q in self.Q]

	def move(self, M, s) :
		self.check_suggestion(M, s)

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

	def check_suggestion(self, M, s) :
		assert isinstance(M, (list)), "M should be a list (of indices)"
		assert (s in [-1, 0, 1]), "s should be a sign: -1, +1 or 0"
		assert (len(M) + len(self.B) <= self.C), "Exceeded the bus capacity"

	def get_w(self) :
		# Number of people waiting in queue, averaged over the stations
		return mean([len(P) for P in self.Q])


