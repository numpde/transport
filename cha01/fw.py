#!/usr/bin/env python3

# R. Andreev, 2016-12-02

# Ensure float division
from __future__ import division

import ai_clock, ai_klock, ai_greedy
import matplotlib.pyplot as plt
import numpy as np
import random as rnd


class World :
	# Deterministic trajectory record
	NEWS = []
	
	C = None # Bus capacity
	N = None # Number of stations
	def __init__(self, C, N) :
		self.C = C
		self.N = N
		
		self.rewind()
	#/def
	
	#public:
	def rewind(self) :
		# Bus location
		self.b = 0
		# Bus passengers
		self.B = []
		# Stations
		self.G = [[] for _ in range(self.N)]
		# Iteration number (time)
		self.i = 0
		# Number of people waiting in queue
		self.w = 0
		
		self.prep_touch()
	#/def
	
	#public:
	def news(self) :
		# Create news
		while (len(self.NEWS) <= self.i) :
			a = rnd.randint(0, self.N-1)
			b = (a + rnd.randint(1, self.N-1)) % self.N
			self.NEWS.append((a, b))
		#/while
		return self.NEWS[self.i]
	#/def
	
	#public:
	def look(self) :
		return self.b, self.B, self.G
	#/def
	
	#public:
	def touch(self, M, s) :
		self.check_suggestion(M, s)
		
		# Passengers mount (in the given order)
		for m in M : self.B.append(self.G[self.b][m])
		# Remove them from the queue
		for m in sorted(M, reverse=True) : self.G[self.b].pop(m)
		# Number of people waiting
		self.w -= len(M)
		
		# Advance bus
		self.b = (self.b + s) % self.N
		
		self.post_touch()
	#/def
	
	#private:
	def post_touch(self) :
		# Advance time
		self.i += 1
		
		self.prep_touch()
		self.check_consistency()
	#/def
	
	#private:
	def prep_touch(self) :
		# Passengers unmount
		self.B = [p for p in self.B if p != self.b]
		
		# New passenger arrives at "a" with destination "b"
		a, b = self.news()
		
		# New passenger in queue
		self.G[a].append(b)
		self.w += 1
	#/def
	
	#private:
	def check_suggestion(self, M, s) :
		assert isinstance(M, (list)), "M should be a list (of indices)"
		assert (s in [-1, 0, 1]), "s should be a sign: -1, +1 or 0"
		assert (len(M) + len(self.B) <= self.C), "Exceeded the bus capacity"
	#/def
	
	#private:
	def check_consistency(self) :
		assert(self.w == sum(len(P) for P in self.G))
	#/def
#/class


class Profiler :
	W = None # Number of people waiting
	
	def __init__(self, wrd, nav, I) :
		assert((0 <= I) and (I <= 1e10))
		
		wrd.rewind()
		assert(wrd.i == 0)
		
		self.W = []
		while (wrd.i < I) :
			wrd.touch(*nav.step(*wrd.look()))
			self.W.append(wrd.w)
		#/while
	#/def
#/class



# Bus capacity
C = 3
# Number of stations
N = 6

# Competing navigators
NAV = [ai_clock.navigator(C, N), ai_klock.navigator(C, N), ai_greedy.navigator(C, N)]

# Ranks (default = 1)
R = [1 for _ in NAV]

# Number of iterations (time steps)
I = 10000

for r in range(len(NAV)-1) :
	# A deterministic world
	wrd = World(C, N)
	
	# Navigator scores (nonnegative; max score loses)
	S = [-1 for _ in NAV]
	for n, nav in enumerate(NAV) :
		if (R[n] != 1) : continue
		
		# Default navigator score = +oo
		S[n] = +np.inf
		
		try :
			print("Profiling:", nav.name)
		except AttributeError as err :
			print("Profiling a no-name navigator")
		#/try
		
		try : 
			# Profile the navigator on the world
			report = Profiler(wrd, nav, I)
			# Score = average number of people waiting
			S[n] = np.mean(report.W)
			plt.plot(report.W)
		except Exception as err :
			print("Error", err)
		#/try
	#/for
	plt.show()
	
	# Rank of the loser
	R[S.index(max(S))] = sum((x == 1) for x in R)
#/for

print("Ranking:")
for r in sorted(R) : 
	print(r, NAV[R.index(r)].name)



