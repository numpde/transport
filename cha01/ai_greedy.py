#!/usr/bin/env python3

class navigator :
	name = "Greedy"
	
	def __init__(self, C, N) :
		self.C = C
		self.N = N

	def step(self, b, B, G) :
		# Number of passengers to board
		z = min(len(G[b]), self.C - len(B))
		# Passenger selection
		M = [m for m in range(z)]
		
		# No passengers?
		if ((not B) and (not M)) : return [], 1
		
		# Next passenger's destination
		t = (B + [G[b][m] for m in M])[0]
		
		# Destination relative to current position
		t = (self.N/2 - ((t-b) % self.N))
		
		# Move towards that destination
		s = (+1) if (t > 0) else (-1)
		
		return M, s


