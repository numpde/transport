#!/usr/bin/env python3

# R. Andreev, 2016-12-02

# Ensure float division
from __future__ import division

import ai_clock, ai_klock, ai_greedy, ai_error, ai_simul

import random, copy
#
class World :

	NEWS = [] # World trajectory record
	C = None  # Bus capacity
	N = None  # Number of stations
	b = None  # Bus position
	B = None  # Bus passengers' destinations
	G = None  # People waiting
	i = None  # Iteration number (time)
	
	def __init__(self, C, N) :
		self.C = C
		self.N = N
		self.rewind()
	#/def
	
	#public:
	def rewind(self) :
		self.b = 0
		self.B = []
		self.G = [[] for _ in range(self.N)]
		self.i = 0
		self.prep_touch()
	#/def
	
	#public:
	def news(self) :
		# Create news if necessary
		while (len(self.NEWS) <= self.i) :
			a = random.randint(0, self.N-1)
			b = (a + random.randint(1, self.N-1)) % self.N
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
		
		# Advance bus
		self.b = (self.b + s) % self.N
		
		self.post_touch()
	#/def
	
	#private:
	def post_touch(self) :
		# Advance time
		self.i += 1
		
		self.prep_touch()
	#/def
	
	#private:
	def prep_touch(self) :
		# Passengers unmount
		self.B = [p for p in self.B if p != self.b]
		
		# New passenger arrives at "a" with destination "b"
		a, b = self.news()
		
		# New passenger in queue
		self.G[a].append(b)
	#/def
	
	#private:
	def check_suggestion(self, M, s) :
		assert isinstance(M, (list)), "M should be a list (of indices)"
		assert (s in [-1, 0, 1]), "s should be a sign: -1, +1 or 0"
		assert (len(M) + len(self.B) <= self.C), "Exceeded the bus capacity"
	#/def
	
	#public:
	def get_w(self) :
		# Number of people waiting in queue
		return sum(len(P) for P in self.G)
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
			self.W.append(wrd.get_w())
		#/while
	#/def
#/class



# Bus capacity
C = 3
# Number of stations
N = 6

# Competing navigators
NAV = [ai_clock.navigator(C, N), ai_klock.navigator(C, N), ai_greedy.navigator(C, N), ai_error.navigator(C, N), ai_simul.navigator(C, N)]

def get_name(nav) :
	try :
		return nav.name
	except :
		return "Unknown"
	#/try
#/def


# Number of iterations (time steps)
I = 10000

# Ranks
R = [None for _ in NAV]



import matplotlib.pyplot as plt
import numpy as np

# While some ranks are undecided
while [r for r in R if r is None] :
	rank = sum((r is None) for r in R)
	print("Number of competitors:", rank)

	# Create a rewindable world
	wrd = World(C, N)
	
	# Navigator scores (nonnegative; max score loses)
	S = [None for _ in NAV]
	for n, nav in enumerate(NAV) :
		if (R[n] is not None) : continue
		
		try :
			print("Profiling:", get_name(nav))
			# Default navigator score = +oo
			S[n] = +np.inf 
			# Profile the navigator on the world
			report = Profiler(wrd, nav, I)
			# Score = average number of people waiting
			S[n] = np.mean(report.W)
			#plt.plot(report.W)
		except Exception as err :
			print("Error:", err)
		#/try
	#/for
	#plt.show()
	
	# Rank the losers
	maxS = max(s for s in S if s is not None)
	for n, s in enumerate(S) : 
		if (s == maxS) : R[n] = rank
	#/for
#/while

print("Final ranking:")
for r in sorted(R) : 
	print(r, get_name(NAV[R.index(r)]))
#/for



