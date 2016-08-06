import numpy as np
import os.path

class State:
	"""
	First all horizontal edges are numbered left to right, top
	to down. Then vertical.
	"""

	# Search depth

	# Size of Grid
	rDots = 0
	cDots = 0
	nEdges = 0

	# Neural network details
	nHidden = 100
	fname = ""
	data = np.empty(0)
	Theta1 = np.empty(0)
	Theta2 = np.empty(0)

	# Change the game details
	@classmethod
	def setGrid(cls, r, c, newData):
		cls.rDots = r
		cls.cDots = c
		cls.nEdges = cls.rDots * (cls.cDots - 1) + cls.cDots * (cls.rDots - 1)
		cls.fname = "data/" + str(cls.rDots) + "_" + str(cls.cDots) + "_" + str(cls.nHidden) + ".npy"
		cls.data = []
		if os.path.isfile(cls.fname) and not newData:
			cls.data = np.load(cls.fname)
		else:
			cls.data = np.random.rand((cls.nEdges + 4)*cls.nHidden + (cls.nHidden+1)*1)

		cls.Theta1 = cls.data[:(cls.nEdges + 4)*cls.nHidden]
		cls.Theta1 = cls.Theta1.reshape((cls.nHidden, (cls.nEdges + 4)))
		cls.Theta2 = cls.data[-(cls.nHidden+1):]
		cls.Theta2 = cls.Theta2.reshape((1, cls.nHidden+1))

	@classmethod
	def save(cls):
		t1 = cls.Theta1.reshape(cls.Theta1.shape[0] * cls.Theta1.shape[1])
		t2 = cls.Theta2.reshape(cls.Theta2.shape[0] * cls.Theta2.shape[1])
		cls.data = np.concatenate((t1, t2))
		np.save(cls.fname, cls.data)

	@classmethod
	def sigmoid(cls, z):
		return 1/(1 + np.exp(-z))

	@classmethod
	def sigmoidGradient(cls, z):
		g = cls.sigmoid(z)
		return g.dot(1 - g)

	@classmethod
	def compute(cls, param):
		z1 = State.Theta1.dot(param)
		z1 = cls.sigmoid(z1)
		z1 = State.Theta2.dot(z1)
		z1 = cls.sigmoid(z1)
		return z1

	#Initialization
	def __init__(self, nTotal, pAg, pOpp, Played):
		self.reset(nTotal, pAg, pOpp, Played)

	def reset(self, nTotal=0, pAg=0, pOpp=0, Played=np.zeros(nEdges)):
		self.nTotal = nTotal
		self.pAg = pAg
		self.pOpp = pOpp
		self.Played = Played

	def add(self, edge, maxi):
		self.nTotal = self.nTotal + 1
		self.Played[edge] = 1

		point = self.isPoint(edge)
		if(point):
			if(maxi):
				self.pAg = self.pAg + point
			else:
				self.pOpp = self.pOpp + point

		return (point == 0)

	def allTrue(self, a, b, c, d):
		if(self.Played[a] > 0.5 and self.Played[b] > 0.5 and self.Played[c] > 0.5 and self.Played[d] > 0.5):
			return 1
		else:
			return 0

	def isPoint(self, edge):
		x = 0
		y = 0
		phase = State.rDots * (State.cDots - 1)
		if(edge < phase):
			if(edge <= phase - State.cDots):
				x = self.allTrue(edge, edge + (State.cDots-1), 
					phase + int(edge/(State.cDots-1))*(State.cDots) + int(edge % (State.cDots-1)),
					phase + int(edge/(State.cDots-1))*(State.cDots) + int(edge % (State.cDots-1)) + 1)

			if(edge >= State.cDots - 1):
				edge = edge - (State.cDots-1)
				y = self.allTrue(edge, edge + (State.cDots-1), 
					phase + int(edge/(State.cDots-1))*(State.cDots) + int(edge % (State.cDots-1)),
					phase + int(edge/(State.cDots-1))*(State.cDots) + int(edge % (State.cDots-1)) + 1)

		else:
			e = edge - phase
			if(e % State.cDots != State.cDots - 1):
				x = self.allTrue(edge, edge + 1, 
					int(e/State.cDots)*(State.cDots-1) + e % State.cDots,
					(int(e/State.cDots)+1)*(State.cDots-1) + e % State.cDots)

			if(e % State.cDots != 0):
				edge = edge - 1
				e = e - 1
				y = self.allTrue(edge, edge + 1, 
					int(e/State.cDots)*(State.cDots-1) + e % State.cDots,
					(int(e/State.cDots)+1)*(State.cDots-1) + e % State.cDots)

		return (x + y)


	def result(self):
		if(self.pAg > self.pOpp):
			return 100
		elif(self.pAg == self.pOpp):
			return 0
		else:
			return -100

	def value(self, maxi, d):
		if(self.nTotal == self.nEdges):
			return self.result()
		elif(d == 0):
			param = np.concatenate(([1], self.Played.copy(), [self.pAg], [self.pOpp], [maxi]))
			return State.compute(param)
		else:
			pass

	def next_move(self, maxi, d):
		if(d == 0):
			print("Some problem.")
		else:
			pass