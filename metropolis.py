#!/usr/bin/env python

'The Ising Model'
'Author: James B Dolan - 12301268'


#--------------------------------------------------------------------
#List of import classes and other objects used in this script
#--------------------------------------------------------------------
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.constants as pc
import random
from random import randint
from math import exp
from math import fsum
import copy
import timeit

'''
#---------------------------------------------
Definitions of constants and the spin lattice
#---------------------------------------------
'''

#Below the Curie temperature, Tc, the system is subcritical, it will demonstrate ferromagnetism. Above the Tc is superccritical, a system will only display paramagnetism at these temperatures. It will lose its own magnetization but respond to a magnetic field. The Tc for iron is 1043K. The atoms have too much energy to align to anything.

#And the heat capacity is Cv = beta/T (<E**2> - <E>**2). Both are determined by fluctuations apparently.
T = float(2)
h = 0.001 # Teslas
beta = 1/T


J = 1.0 #interaction energy

h = 1	#external magnetic field
#if J > 0 the interaction is ferromagnetic ~ spins line up
#if J < 0 the interaction is antiferromagnetic ~ checkerboard
#if h > 0 the spin site j desires to line up in the positive direction => +equilibrium
#if h < 0 the spin site j desires to line up in the negative direction => -equilibrium

n = int(raw_input("n = ")) #The size of the square matrix. Requires user input
print 'Expect a runtime of approximately: ', (n**2 + n)/425, 'seconds.'


spin_lattice = np.zeros((n,n), dtype=int) #Creation of the spin lattice

#Sets a random array of 1 and -1	
for i in range(n):
	for j in range(n):
		k = random.random()
		if k > 0.5:
 			k = 1
		else:
			k = -1
		spin_lattice[i][j] = k
		
#M is the magnetization of the material in other words the magnetic dipole moment per unit volume, measured in A/m. It is the mean of the spin lattice.



'''
#-----------------------------
Definitions of the functions
#-----------------------------
'''

#These 4 functions, given an row number, i, and column number, j, will produce a neighbour either in the direction right, left, up or down. They are called in the following function, pairs(i,j)
def Sj_r(i, j):
	Sj_r = spin_lattice[i][j+1]
	return Sj_r
def Sj_l(i, j):	
	Sj_l = spin_lattice[i][j-1]
	return Sj_l
def Sj_u(i, j):	
	Sj_u = spin_lattice[i-1][j]
	return Sj_u
def Sj_d(i, j):	
	Sj_d = spin_lattice[i+1][j]
	return Sj_d

#As stated above, this function uses the above 4 functions. It breaks down a 2D matrix into 9 locations. It creates a list of neighbours. It sums them and counts the number of them in the list.
def pairs(i, j):
	Si = spin_lattice[i][j]

	# Only considering the top row
	if i == 0:
		if j == 0:	#Top left corner
			Sj = [Sj_r(i, j), Sj_d(i, j)]
		if j == n-1:	#Top right corner
			Sj = [Sj_l(i, j), Sj_d(i, j)]
		elif j != 0 and j != n-1:	#Elsewhere in that top row
			Sj = [Sj_r(i, j), Sj_l(i, j), Sj_d(i, j)]
	
	# Only considering bottom row
	if i == n-1:
		if j == 0:	#Bottom left corner
			Sj = [Sj_r(i, j), Sj_u(i, j)]
		if j == n-1:	#Bottom right corner
			Sj = [Sj_l(i, j), Sj_u(i, j)]			
		elif j != 0 and j != n-1:	#Elsewhere in that bottom row
			Sj = [Sj_r(i,j), Sj_l(i,j), Sj_u(i,j)]

	# Only considering the left column but between the top and bottom row.
	if j == 0 and i > 0 and i < n-1:
		Sj = [Sj_r(i,j), Sj_d(i,j), Sj_u(i,j)]
	
	# Only considering the right column but between the top and bottom row.
	if j == n-1 and i > 0 and i < n-1:
		Sj = [Sj_l(i,j), Sj_u(i,j), Sj_d(i,j)]
	
	# Only considering the elements inside the largest square of elements
	if n-1 > i > 0 and n-1 > j > 0:
		Sj = [Sj_r(i,j), Sj_l(i,j), Sj_u(i,j), Sj_d(i,j)]
		
	num_of_Sj = len(Sj)
	Sj_sum = sum(Sj)
	return Si, num_of_Sj, Sj_sum

#Definition of the energy E calculation, returning the calculated value
def H(i,j):
	Si = pairs(i,j)[0]
	num_of_Sj = pairs(i,j)[1]
	Sj_sum = pairs(i,j)[2]
	H = Si * ( -J * (Sj_sum) - num_of_Sj*h )
	return H, Si, num_of_Sj, Sj_sum
	
#Calculation of the Magnetic Susceptibility, Xv
def mag_sus( spin_lattice ):
	global beta
	#On page 42 of Hutzler's lectures, Xv is defined by beta * (<M**2> - <M>**2)
	Xv = beta * ( np.mean( spin_lattice**2 ) - np.mean( spin_lattice )**2 )
	return Xv
	
#This copy function is to keep the original lattice for comparison with the equilibrium.
original_spin_lattice = copy.deepcopy(spin_lattice) #http://bit.ly/1P0AC9u
'''
#---------------------------------------
#Iterative process toward Equilibrium 
#---------------------------------------
'''
#This start the runtime timer
start = timeit.default_timer() #http://bit.ly/1mUNYrh
	
#Loop counter is used to get an approximation to the number of sweeps carried out
loop_counter = 0.0
enough = 300
M_arr = []
mag_sus_arr = []
	
	#The below while loop reads "while the lattice is not ferromagnetic do:", ferromagnetic meaning, all elements point in the same direction
while abs( np.mean(spin_lattice) ) != 1: 
	#while loop_counter < enough:
 	# 1. Pick a spin site randomly and calculate the contribution to the energy involving this spin.
	i = randint( 0, n-1 )	#Selection probability is hopefully catered for here.
	j = randint( 0, n-1 )	#Otherwise "ergodicity" wouldn't be met!
	H1 = H(i,j)[0]
	# 2. Calculate the energy for the flipped original spin.
	spin_lattice[i][j] *= -1 #spin value flipped
	H2 = H(i,j)[0]			 
	dE = H1 - H2	#	dE = Hij_flip - Hij. HE = sum of all Hij
	P_flip = exp( -beta*dE )
	if dE <= 0: 
		spin_lattice[i][j] *= -1
		# 3. If the new energy is less, keep the flipped value.
	ran_num = random.random()
	if dE > 0 and ran_num < P_flip:	#Acceptance probability
		spin_lattice[i][j] *= -1
		# 4. If the new energy is more, only keep with probability P_flip
	
		#Addition of the net magnetization and magnetic susceptibility to its list
		#M_arr.append( np.mean(spin_lattice) )
		#M = np.mean(spin_lattice)
		#mag_sus_arr.append( mag_sus( spin_lattice ) )
		
	eqlb_M = np.mean(spin_lattice)	#Equilibrium value for magnetization
	eqlb_mag_sus = mag_sus( spin_lattice )	#Equilibrium value for magnetic sus.
	loop_counter += 1
	

#------------------------------
#Results to output in terminal
#------------------------------

#This is only an approximate because i and j are generated randomly, so for 25 elements, n = 5, in 25 loops it is quite likely that not every element was evaluated.
approx_sweeps = loop_counter/(n)**2
	
	#Stops the runtime timer
stop = timeit.default_timer()
runtime = stop - start

#The "%0.3f" %approx_sweeps bit just rounds off the number of sweeps to 3 sig figs.
print 'Runtime = ', runtime, 'seconds for a total of ~', "%0.3f" %approx_sweeps, 'sweeps.'


'''
#--------------------------------------------------------------------
Everything below is for plotting
#--------------------------------------------------------------------
'''
#------------------------------
#Plotting the statistical info
#------------------------------
#fig1, (ax3, ax4) = plt.subplots(ncols=2) #http://bit.ly/1mXPklK

#x1 = np.arange(0, loop_counter, 1)
#x2 = np.arange(0, loop_counter, 1)
#fit = 1 - np.exp(-x1/(25*n + n**2))
#plt.subplot(1)(x1, M_arr, 'b', x1, fit, 'r')
#plt.plot(x2, mag_sus_arr, 'g')
#plt.ylim( -3e9,3e9 )
#plt.xlim( 0, 3000)
'''So I'm getting ambiguous plots for mag_sus. sometimes it briefly oscillates about x axis, sometimes it increase from minus 2e8 to 0, and lastly sometimes it decrease from 2e8 to 0. The going to zero makes sense because the lattice is becomin ferromagnetic. The randomness of the plots though is bizarre' '''

#-----------------------------
#Plotting the spin lattices
#-----------------------------

#Figure with sub-plots
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(7,3.5)) #http://bit.ly/1mXPklK

#Move plots over to make room for the colour bar axis, which is defined there below
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.19, 0.02, 0.62])

#The colours I want my plots to be in. Grey for 1, and black for -1. Grey and black is easier on the eyes.
colour_map = mpl.colors.ListedColormap(['black', 'grey'])
bounds = [-1, 0, 1]
norm = mpl.colors.BoundaryNorm(bounds, colour_map.N)

#Generation of colour maps and then of the colour bar
ax1.imshow(original_spin_lattice, interpolation='nearest', cmap = colour_map, norm = norm)
ax1.set_title('Before')
img = ax2.imshow(spin_lattice, interpolation='nearest', cmap = colour_map, norm = norm)
ax2.set_title('After')
plt.colorbar(img, cax=cbar_ax, cmap=colour_map)

plt.show()

'''
#--------------------------------------------------------------------
End of script
#--------------------------------------------------------------------
'''