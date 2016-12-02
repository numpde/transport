#!/usr/bin/env python3

class navigator :
	name = "Clock"
	
	def __init__(self, C, N) :
		self.C = C
		self.N = N
	#/def

	def step(self, b, B, G) :
		# Number of passengers to board
		z = min(len(G[b]), self.C - len(B))
		# Passenger selection
		M = [m for m in range(z)]
		
		return M, +1
	#/def
#/class

