#!/usr/bin/env python3

import ai_clock, ai_klock, ai_greedy
import matplotlib.pyplot as plt
import numpy as np
import random as rnd


# Bus capacity
C = 3
# Number of iterations
I = 10000
# Number of stations
N = 6

# Competitors
NAV = [ai_clock.navigator(C, I, N), ai_klock.navigator(C, I, N), ai_greedy.navigator(C, I, N)]



class Scenario :
	def __init__(self, C, I, N) :
		self.A = [] # Location
		self.B = [] # Destination
		for _ in range(I) :
			a = rnd.randint(0, N-1)
			b = (a + rnd.randint(1, N-1)) % N
			self.A.append(a)
			self.B.append(b)
		#/for
	#/def
#/class

def profile(nav, sce) :
	# History (number of people waiting)
	h = []
	# Bus location
	b = 0
	# Bus passengers
	B = []
	# Route
	G = [[] for _ in range(N)]
	
	for i in range(I) :
		# Passengers unmount
		B = [p for p in B if p != b]
		
		# New passenger
		G[sce.A[i]].append(sce.B[i])
		
		# Ask the navigator
		M, s = nav.step(i, b, B, G)
		assert isinstance(M, (list)) and (s in [-1, 0, 1]), "navigator.step should return a list and a sign"
		assert (len(M) + len(B) <= C), "navigator.step exceeded the bus capacity"
		
		# Passengers mount (in the given order)
		for m in M : B.append(G[b][m])
		# Remove them from the waiting list
		M.sort(reverse=True)
		for m in M : G[b].pop(m)
		
		# Advance bus
		b = (b + s) % N
		
		# History
		h.append(sum([len(P) for P in G]))
	#/for
	
	return h
#/def


# Ranks (default = 1)
R = [1 for _ in NAV]

for r in range(len(NAV)-1) :
	sce = Scenario(C, I, N)
	
	# Scores (smaller is better)
	S = [0 for _ in NAV]
	for n, nav in enumerate(NAV) :
		if (R[n] != 1) : continue
		# Default score = +oo
		S[n] = +np.inf
		try : 
			print("Running", nav.name)
			# Returns the history of people waiting
			h = profile(nav, sce)
			# Score = average number of people waiting
			S[n] = sum(h) / len(h)
			plt.plot(h)
		except AttributeError as err :
			print("Error", err)
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


