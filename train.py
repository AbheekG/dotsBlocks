from timeit import default_timer as timer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from state import State

#Size of Grid
rows = int(input('Enter number of dots rows: '))
cols = int(input('Enter number of dots columns: '))
if(rows < cols):
	rows, cols = cols, rows
if(rows < 1 | cols < 1):
	print('Please enter valid grid.')
else:
	State.setGrid(rows, cols, False)

# iterations = int(input('Enter number of iterations: '))
# alpha = float(input('Enter alpha: '))
iterations = 1000000
alpha = 0.02
# lamda = 0
# J = []


def descent(x, game, pos, maxi):
	beta = State.nEdges - pos
	maxi = maxi ^ (x.add(game[pos], maxi))
	a1 = np.concatenate(([1], x.Played.copy(), [x.pAg], [x.pOpp], [maxi]))
	#print(x.Played)

	if(beta == 1):
		y = x.result()
		#print(game, y)
	else:
		y = descent(x, game, pos + 1, maxi)
		y = 0.5 + (y - 0.5) / (beta - 1)

	a2 = State.Theta1.dot(a1)
	a2 = np.concatenate(([1], State.sigmoid(a2)))
	h = State.Theta2.dot(a2)
	h = State.sigmoid(h)

	#print(y, h)
	# global J
	# if(beta == 10):
	# 	J = J + [-y*np.log(h) - (1-y)*np.log(1-h)]

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

# axes = plt.gca()
# axes.set_ylim([0,1])
# plt.plot(range(len(J)), J)
# plt.show()

print(State.Theta1, State.Theta2)
State.save()
