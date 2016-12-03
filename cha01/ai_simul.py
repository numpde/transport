#!/usr/bin/env python3

import random as rnd

class navigator :
	name = "Simul"
	
	def __init__(self, C, N) :
		self.C = C
		self.N = N

	def step(self, b, B, Q) :
		# Number of passengers to board
		z = min(len(Q[b]), self.C - len(B))
		# Passenger selection
		M = [m for m in range(z)]
		
		return M, rnd.choice([-1, +1])

