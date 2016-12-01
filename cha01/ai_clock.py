#!/usr/bin/env python3

# AI part

class navigator :
	name = "Clock"
	
	def __init__(self, C, I, N) :
		self.C = C
		self.I = I
		self.N = N
	#/def

	def step(self, i, b, B, G) :
		# Number of passengers to board
		z = min(len(G[b]), self.C - len(B))
		# Passenger selection
		M = [m for m in range(z)]
		
		return M, +1
	#/def
#/class

