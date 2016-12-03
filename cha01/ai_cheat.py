#!/usr/bin/env python3

import ai_clock

class navigator :
	name = "Cheat"
	
	def __init__(self, C, N) :
		self.AI = ai_clock.navigator(C, N)

	def step(self, b, B, Q) :
	
		# Try to cheat:
		for n in range(len(Q)) :
			Q[n] = []
		
		return self.AI.step(b, B, Q)


