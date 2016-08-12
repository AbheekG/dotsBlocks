from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt

from state import State
from printer import gamePrinter

# Size of Grid
# rows = int(input('Enter number of dots rows: '))
# cols = int(input('Enter number of dots columns: '))
# if(rows < 1 | cols < 1):
# 	print('Please enter valid grid.')
# else:
# 	State.setGrid(rows, cols, False)

State.setGrid(5, 5, False)

# Depth, here for each level of depth ony one player moves
time_lim = 1
depth = 3

# define start state
play = State(0, 0, 0, np.zeros(State.nEdges))
game = np.zeros(State.nEdges)

# You start the game
print('Play the game by entering the number for the edge.')
maxi = False
while(play.nTotal < State.nEdges):
	start = timer()
	if(maxi):
		move = play.next_move(maxi, int(depth))
	else:
		move = int(input('Please enter your move: '))

	game[play.nTotal] = move
	maxi = maxi ^ (play.add(move, maxi))
	# temp = abs((State.nEdges - play.nTotal)/(State.nEdges - play.nTotal - depth))
	# if(temp > 1):
	#depth = depth * 1.035

	print(chr(27) + "[2J")
	gamePrinter(play.Played, play.Score, State.rDots, State.cDots)
	
	print('\n\nThe move was: ', move)
	print('Depth = ', depth)
	#print(play.Score)
	#print('Game = ', game)

	end = timer()
	if(end - start < time_lim):
		depth = depth + 1

if(play.result() == 1):
	print('Sorry. You lost.')
elif(play.result() == 0.5):
	print('Draw. Just little more effort.')
else:
	print('Congratulations. You won.')