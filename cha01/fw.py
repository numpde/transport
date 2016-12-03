#!/usr/bin/env python3

# R. Andreev, 2016-12-03


# Bus capacity
C = 3
# Number of stations
N = 6


from world import World
from numpy import mean
#
class Profiler :
	def __init__(self, wrd, nav, I) :
		# W[i] = average number of people waiting at time i
		self.W = []   
		# w = average over time
		self.w = None

		assert((0 < I) and (I <= 1e9))

		wrd.rewind()
		assert(wrd.i == 0)

		while (wrd.i < I) :
			wrd.move(*nav.step(*wrd.look()))
			self.W.append(wrd.get_w())
		
		assert(len(self.W))
		self.w = mean(self.W)



#
print("1. Initializing navigators")

# Competing navigators
NAV = []

# Load all local ai_* modules
# Ref: http://stackoverflow.com/questions/8718885/
from os.path import dirname, basename, isfile
import glob, importlib, sys
modules = glob.glob(dirname(__file__) + "/*.py")
modules = [basename(f)[:-3] for f in modules if isfile(f)]
#modules = list(set(modules)) # unique names
modules = [m for m in modules if m.startswith("ai_")]
for module in modules : 
	try :
		m = importlib.import_module(module)
		Nav = getattr(m, 'navigator')
		NAV.append(Nav(C, N))
		print("Loaded module", module)
	except Exception as err:
		print("Loading module", module, "failed:", err)

def get_name(nav) :
	try :
		return nav.name
	except :
		return "Unknown"



#
print("2. Profiling navigators")

# Ranks
R = [None for _ in NAV]
# Scores
S = [[] for _ in NAV]

#import matplotlib.pyplot as plt
from numpy import mean

# While some ranks are undecided
while [r for r in R if r is None] :
	rank = sum((r is None) for r in R)
	print("Number of competitors:", rank)

	# Create a rewindable world
	wrd = World(C, N)

	# Navigator scores (nonnegative; max score loses)
	K = []
	for n, nav in enumerate(NAV) :
		if (R[n] is not None) : continue
		
		print(" - Profiling:", get_name(nav))
		try :
			# Number of iterations (time steps)
			I = 10000
			# Profile the navigator on the world
			report = Profiler(wrd, nav, I)
			# Score = average number of people waiting
			K.append((n, report.w))
			#plt.plot(report.W) # History
		except Exception as err :
			R[n] = rank
			print("    - Error:", err)
	
	
	
	# Rank the losers
	for n, s in K :
		if (s == max(s for n, s in K)) : R[n] = rank
		S[n].append(mean(S[n] + [s]))


#
print("3. Final ranking:")

for r in sorted(list(set(R))) :
	print(r, [get_name(NAV[i]) for i, rr in enumerate(R) if (r == rr)])


import matplotlib.pyplot as plt
for s in S :
	plt.plot(s, '-x')
plt.yscale('log')
plt.xlabel('Round')
plt.ylabel('Score (less is better)')
plt.legend([get_name(nav) for nav in NAV], numpoints=1)
plt.show()


