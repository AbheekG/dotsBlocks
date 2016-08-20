from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt

from state import State
from printer import gamePrinter

#Size of Grid
rows = int(input('Enter number of dots rows: '))
cols = int(input('Enter number of dots columns: '))
if(rows < cols):
	rows, cols = cols, rows
if(rows < 1 | cols < 1):
	print('Please enter valid grid.')
else:
	State.setGrid(rows, cols, False)

# Set depth and approximate time limit for move.
time_lim = 1
depth = 3

# define start state
play = State(0, 0, 0, np.zeros(State.nEdges))
game = np.zeros(State.nEdges)

# Start the game
print('Play the game by entering the number for the edge.')
maxi = False
while(play.nTotal < State.nEdges):
	start = timer()
	# Get move
	if(maxi):
		move = play.next_move(maxi, int(depth))
	else:
		move = int(input('Please enter your move: '))

	game[play.nTotal] = move
	maxi = maxi ^ (play.add(move, maxi))

	# Print game status
	print(chr(27) + "[2J")
	gamePrinter(play.Played, play.Score, State.rDots, State.cDots)
	
	print('\n\nThe move was: ', move)
	print('Depth = ', depth)
	#print(play.Score)
	#print('Game = ', game)

	end = timer()
	print('Time taken for move = ', end - start)
	if(end - start < time_lim):
		depth = depth + 1

print('The game play was: ', game)
if(play.result() == 1):
	print('Sorry. You lost.')
elif(play.result() == 0.5):
	print('Draw. Just little more effort.')
else:
	print('Congratulations. You won.')