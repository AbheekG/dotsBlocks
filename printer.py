from timeit import default_timer as timer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from state import State

def gamePrinter(A, B, r, c):
	phase = r * (c - 1)
	for a in range(r-1):

		for b in range(c-1):
			print('o ', end='')
			i = (c-1)*a + b
			if(A[i] > 0):
				print('----- ', end='')
			else:
				print(' ', i, end='')
				if(i < 10):
					print('   ', end='')
				else:
					print('  ', end='')
		print('o')

		for b in range(c):
			i = phase + a*c + b
			if(A[i] > 0):
				print('|       ', end='')
			else:
				print('        ', end='')
		print('')

		for b in range(c):
			i = phase + a*c + b
			if(A[i] > 0):
				print('| ', end='')
			else:
				print(i, end='')
				if(i < 10):
					print(' ', end='')
			print('  ', end='')
			if(b < c-1):
				if(B[a*(c-1) + b] > 0):
					print('C', end='')
				elif(B[a*(c-1) + b] < 0):
					print('A', end='')
				else:
					print(' ',end='')
			else:
				print(' ',end='')
			print('   ', end='')
		print('')

		for b in range(c):
			i = phase + a*c + b
			if(A[i] > 0):
				print('|       ', end='')
			else:
				print('        ', end='')
		print('')

	for b in range(c-1):
		print('o ', end='')
		i = (c-1)*(r-1) + b
		if(A[i] > 0):
			print('----- ', end='')
		else:
			print(' ', i, end='')
			if(i < 10):
				print('   ', end='')
			else:
				print('  ', end='')
	print('o')