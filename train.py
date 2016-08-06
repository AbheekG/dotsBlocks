from timeit import default_timer as timer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from state import State

State.setGrid(4, 4, False)
iterations = 100000
alpha = 0.01
lamda = 0


def descent(x, game, pos, maxi):
	beta = State.nEdges - pos
	maxi = maxi ^ (x.add(game[pos], maxi))
	a1 = np.concatenate(([1], x.Played.copy(), [x.pAg], [x.pOpp], [maxi]))
	#print(x.Played)

	if(beta == 1):
		y = State.sigmoid(x.result())
		#print(game, y)
	else:
		y = descent(x, game, pos + 1, maxi)
		y = 0.5 + (y - 0.5) / (beta - 1)

	#print(y, maxi)

	a2 = State.Theta1.dot(a1)
	a2 = np.concatenate(([1], State.sigmoid(a2)))
	h = State.Theta2.dot(a2)
	h = State.sigmoid(h)

	#print(y, h)

	delta3 = h - y
	delta2 = State.Theta2.transpose() * delta3
	delta2 = delta2[1:,0] * a2[1:] * (1 - a2[1:])

	delta2 = delta2.reshape(delta2.shape[0], 1)
	a1 = a1.reshape(1, a1.shape[0])
	a2 = a2.reshape(1, a2.shape[0])
	State.Theta1 = State.Theta1 - alpha*delta2.dot(a1)
	State.Theta2 = State.Theta2 - alpha*delta3 * a2

	return y

for i in range(iterations):
	if(i % 5000 == 0):
		print(i)#, State.Theta1, State.Theta2)
		pass
	game = np.random.permutation(State.nEdges)
	start = State(0, 0, 0, np.zeros(State.nEdges))
	descent(start, game, 0, True)

State.save()