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
	def setGrid(cls, r=4, c=4, newData=False):
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
		z1 = np.concatenate(([1], cls.sigmoid(z1)))
		z1 = State.Theta2.dot(z1)
		z1 = cls.sigmoid(z1)
		return z1

	#Initialization
	def __init__(self, nTotal=0, pAg=0, pOpp=0, Played=[]):
		self.reset(nTotal, pAg, pOpp, Played)

	def reset(self, nTotal=0, pAg=0, pOpp=0, Played=[]):
		self.nTotal = nTotal
		self.pAg = pAg
		self.pOpp = pOpp
		self.Played = Played
		self.Score = np.zeros((State.rDots - 1) * (State.cDots - 1))

	def add(self, edge, maxi):
		if(self.Played[edge] == 1):
			print('invalid move')
			return False
		self.nTotal = self.nTotal + 1
		self.Played[edge] = 1

		point = self.isPoint(edge, maxi)
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

	def isPoint(self, edge, maxi):
		x = 0
		y = 0
		phase = State.rDots * (State.cDots - 1)
		if(edge < phase):
			if(edge <= phase - State.cDots):
				x = self.allTrue(edge, edge + (State.cDots-1), 
					phase + int(edge/(State.cDots-1))*(State.cDots) + int(edge % (State.cDots-1)),
					phase + int(edge/(State.cDots-1))*(State.cDots) + int(edge % (State.cDots-1)) + 1)
				if(x == 1):
					self.Score[edge] = 1 if maxi else -1

			if(edge >= State.cDots - 1):
				edge = edge - (State.cDots-1)
				y = self.allTrue(edge, edge + (State.cDots-1), 
					phase + int(edge/(State.cDots-1))*(State.cDots) + int(edge % (State.cDots-1)),
					phase + int(edge/(State.cDots-1))*(State.cDots) + int(edge % (State.cDots-1)) + 1)
				if(y == 1):
					self.Score[edge] = 1 if maxi else -1

		else:
			e = edge - phase
			if(e % State.cDots != State.cDots - 1):
				x = self.allTrue(edge, edge + 1, 
					int(e/State.cDots)*(State.cDots-1) + e % State.cDots,
					(int(e/State.cDots)+1)*(State.cDots-1) + e % State.cDots)
				if(x == 1):
					self.Score[int(e/State.cDots)*(State.cDots-1) + e % State.cDots] = 1 if maxi else -1

			if(e % State.cDots != 0):
				edge = edge - 1
				e = e - 1
				y = self.allTrue(edge, edge + 1, 
					int(e/State.cDots)*(State.cDots-1) + e % State.cDots,
					(int(e/State.cDots)+1)*(State.cDots-1) + e % State.cDots)
				if(y == 1):
					self.Score[int(e/State.cDots)*(State.cDots-1) + e % State.cDots] = 1 if maxi else -1

		return (x + y)


	def result(self):
		if(self.pAg > self.pOpp):
			return 1
		elif(self.pAg == self.pOpp):
			return 0.5
		else:
			return 0

	def value(self, maxi, d):
		val = 0 if maxi else 1
		if(d == 0):# or (maxi and d == 1)):
			param = np.concatenate(([1], self.Played, [self.pAg], [self.pOpp], [maxi]))
			val = State.compute(param)
		elif(self.nTotal == State.nEdges):
			return self.result()
		else:
			x = State(self.nTotal, self.pAg, self.pOpp, self.Played.copy())
			for i in range(State.nEdges):
				if(x.Played[i] == 1):
					continue
				else:
					temp_maxi = maxi ^ x.add(i, maxi)
					#print('State =',self.nTotal,'Depth =',d,',Value =',val,', Next move =',i,", Maxi =",maxi)
					if(maxi):
						val = max(val, x.value(temp_maxi, d-1))
					else:
						val = min(val, x.value(temp_maxi, d-1))
					
					x.Played[i] = 0
					x.pAg = self.pAg
					x.pOpp = self.pOpp
					x.nTotal = self.nTotal
		
		#print(val)
		return val

	def next_move(self, maxi, d):
		x = State(self.nTotal, self.pAg, self.pOpp, self.Played.copy())
		move = -1
		max_value = -1
		for i in range(State.nEdges):
			#print('Checking...', i)
			if(x.Played[i] == 1):
				continue
			else:
				temp_maxi = maxi ^ x.add(i, maxi)
				val = x.value(temp_maxi, d-1)
				if(val > max_value):
					#print('State =',self.nTotal,'Depth =',d,',Value =',val,', Next move =',i,", Maxi =",maxi)
					move = i
					max_value = val
				
				x.Played[i] = 0
				x.pAg = self.pAg
				x.pOpp = self.pOpp
				x.nTotal = self.nTotal

		#print('Move = ', move, ", Max value =", max_value)
		if(move < 0 or max_value < 0):
			print('Error. No move.')

		return move	