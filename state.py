import numpy as np
import os.path

class State:
	"""
	First all horizontal edges are numbered left to right, top
	to down. Then vertical.
	"""

	# Search depth

	# Size of Grid
	rDots = 4
	cDots = 4
	nEdges = rDots * (cDots - 1) + cDots * (rDots - 1)

	# Neural network details
	nHidden = 100
	fname = "data/" + str(rDots) + "_" + str(cDots) + "_" + str(nHidden) + ".npy"
	data = []
	if os.path.isfile(fname):
		data = np.load(fname)
	else:
		data = np.random.rand(nEdges*nHidden + nHidden*1)

	Theta1 = data[:nEdges*nHidden]
	Theta1 = Theta1.reshape((nHidden, nEdges))
	Theta2 = data[-nHidden:]
	Theta2 = Theta2.reshape((1, nHidden))

	def __init__(self, nTotal, nAg, nOpp, Ag, Opp):
		self.reset(nTotal, nAg, nOpp, Ag, Opp)

	def reset(self, nTotal, nAg, nOpp, Ag, Opp):
		self.nTotal = nTotal
		self.nAg = nAg
		self.nOpp = nOpp
		self.Ag = Ag
		self.Opp = Opp

	def value(self, isMax, d):
		if(nTotal = self.nEdges):
			return Utility()
		elif(d == 0):
			param = np.zeros(nEdges)
			for i in Ag:
				param[i] = 1
			for i in Opp:
				param[i] = -1
			# TODO: use neural networks
			return Weights.dot(param)
		else:


	def next_move(self, isMax, d):
		if(d == 0):
			print("Some problem.")
		else: