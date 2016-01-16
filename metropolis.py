#!/usr/bin/env python

'The Ising Model'
'Author: James B Dolan - 12301268'


#--------------------------------------------------------------------
#List of import classes and other objects used in this script
#--------------------------------------------------------------------
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg') #Changing the 'backend' of mpl is necessary to make it do different things. I need TkAgg for the animation of the spin lattices for the range of temps.
import matplotlib.pyplot as plt
import scipy.constants as pc
import random
from random import randint
from math import exp, sqrt
from math import fsum
import copy
import timeit
import time
'''
#---------------------------------------------
Definitions of constants and the spin lattice
#---------------------------------------------
'''

#Below the Curie temperature, Tc, the system is subcritical, it will demonstrate ferromagnetism. Above the Tc is superccritical, a system will only display paramagnetism at these temperatures. It will lose its own magnetization but respond to a magnetic field. The Tc for iron is 1043K. The atoms have too much energy to align to anything.

#And the heat capacity is Cv = beta/T (<E**2> - <E>**2). Both are determined by fluctuations apparently.

temperatures = np.arange(1.0,50.0,1.0) # Kelvin

#kb = pc.k * (5*(10**21))
#print kb
beta_array = [1/T for T in temperatures]

J = 1.0 #interaction energy

h = 1	#external magnetic field in Teslas
#if J > 0 the interaction is ferromagnetic ~ spins line up
#if J < 0 the interaction is antiferromagnetic ~ checkerboard
#if h > 0 the spin site j desires to line up in the positive direction => +equilibrium
#if h < 0 the spin site j desires to line up in the negative direction => -equilibrium
n=45
#n = int(raw_input("n = ")) #The size of the square matrix. Requires user input
#print 'Expect a runtime of approximately: ', (n**2 + n)/425, 'seconds.'
start_program_timer = timeit.default_timer() #http://bit.ly/1mUNYrh


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

#This copy function is to keep the original lattice for comparison with the equilibrium.
#original_spin_lattice = copy.deepcopy(spin_lattice) #http://bit.ly/1P0AC9u
spin_lattice_copies = [copy.deepcopy(spin_lattice) for i in temperatures]

sweep_limits = np.arange(1,40,1)

'''
#-----------------------------
Definitions of the functions
#-----------------------------
'''

#These 4 functions, given an row number, i, and column number, j, will produce a neighbour either in the direction right, left, up or down. They are called in the following function, pairs(i,j)
def Sj_r(spin_lattice_copy, i, j):
	Sj_r = spin_lattice_copy[i][j+1]
	return Sj_r
def Sj_l(spin_lattice_copy, i, j):
	Sj_l = spin_lattice_copy[i][j-1]
	return Sj_l
def Sj_u(spin_lattice_copy, i, j):
	Sj_u = spin_lattice_copy[i-1][j]
	return Sj_u
def Sj_d(spin_lattice_copy, i, j):
	Sj_d = spin_lattice_copy[i+1][j]
	return Sj_d

#As stated above, this function uses the above 4 functions. It breaks down a 2D matrix into 9 locations. It creates a list of neighbours. It sums them and counts the number of them in the list.
def pairs(spin_lattice_copy, i, j):
	Si = spin_lattice_copy[i][j]

	# Only considering the top row
	if i == 0:
		if j == 0:	#Top left corner
			Sj = [Sj_r(spin_lattice_copy, i, j), Sj_d(spin_lattice_copy, i, j)]
		if j == n-1:	#Top right corner
			Sj = [Sj_l(spin_lattice_copy, i, j), Sj_d(spin_lattice_copy, i, j)]
		elif j != 0 and j != n-1:	#Elsewhere in that top row
			Sj = [Sj_r(spin_lattice_copy, i, j), Sj_l(spin_lattice_copy, i, j), Sj_d(spin_lattice_copy, i, j)]

	# Only considering bottom row
	if i == n-1:
		if j == 0:	#Bottom left corner
			Sj = [Sj_r(spin_lattice_copy, i, j), Sj_u(spin_lattice_copy, i, j)]
		if j == n-1:	#Bottom right corner
			Sj = [Sj_l(spin_lattice_copy, i, j), Sj_u(spin_lattice_copy, i, j)]
		elif j != 0 and j != n-1:	#Elsewhere in that bottom row
			Sj = [Sj_r(spin_lattice_copy, i,j), Sj_l(spin_lattice_copy, i,j), Sj_u(spin_lattice_copy, i,j)]

	# Only considering the left column but between the top and bottom row.
	if j == 0 and i > 0 and i < n-1:
		Sj = [Sj_r(spin_lattice_copy, i,j), Sj_d(spin_lattice_copy, i,j), Sj_u(spin_lattice_copy, i,j)]

	# Only considering the right column but between the top and bottom row.
	if j == n-1 and i > 0 and i < n-1:
		Sj = [Sj_l(spin_lattice_copy, i,j), Sj_u(spin_lattice_copy, i,j), Sj_d(spin_lattice_copy, i,j)]

	# Only considering the elements inside the largest square of elements
	if n-1 > i > 0 and n-1 > j > 0:
		Sj = [Sj_r(spin_lattice_copy, i,j), Sj_l(spin_lattice_copy, i,j), Sj_u(spin_lattice_copy, i,j), Sj_d(spin_lattice_copy, i,j)]

	num_of_Sj = len(Sj)
	Sj_sum = sum(Sj)
	return Si, num_of_Sj, Sj_sum


#Definition of the energy E calculation, returning the calculated value
def H(spin_lattice_copy, i,j):
	Si = pairs(spin_lattice_copy, i,j)[0]
	num_of_Sj = pairs(spin_lattice_copy, i,j)[1]
	Sj_sum = pairs(spin_lattice_copy, i,j)[2]
	H = Si * ( -J * (Sj_sum) - num_of_Sj*h )
	return H, Si, num_of_Sj, Sj_sum

#Calculation of the Magnetic Susceptibility, Xv
def mag_sus( spin_lattice_copy, beta ):
	#On page 42 of Hutzler's lectures, Xv is defined by beta * (<M**2> - <M>**2)
	Xv = beta * ( np.mean( spin_lattice_copy**2 ) - (np.mean( spin_lattice_copy ))**2 )
	return Xv

#Calculation of the energy change and acceptance of the flip in each iteration of the process toward equilibrium

def iteration(spin_lattice_copy, beta):
	global n
	print 'Starting on this temperature:',(1/beta),'K'

	sweep_limit = 60
	#This start the runtime timer
	start_iterations_timer = timeit.default_timer() #http://bit.ly/1mUNYrh

	#Loop counter is used to get an approximation to the number of sweeps carried out
	loop_counter = 0.0
	sweeps = 0.0
	#M_arr = []
	#mag_sus_arr = []
	#The below while loop reads "while the lattice is not ferromagnetic do:", ferromagnetic meaning, all elements point in the same direction
	#while abs( np.mean(spin_lattice) ) != 1:
	while sweeps < sweep_limit: #120 sweeps for T=200K, 13 for T=20K, 6 for T=2K
 		# 1. Pick a spin site randomly and calculate the contribution to the energy involving this spin.
		i = randint( 0, n-1 )	#Selection probability is hopefully catered for here.
		j = randint( 0, n-1 )	#Otherwise "ergodicity" wouldn't be met!
		H1 = H(spin_lattice_copy, i,j)[0]
		# 2. Calculate the energy for the flipped original spin.
		spin_lattice_copy[i][j] *= -1 #spin value flipped
		H2 = H(spin_lattice_copy, i,j)[0]
		dE = H1 - H2	#	dE = Hij_flip - Hij. HE = sum of all Hij
		P_flip = exp( -beta*dE )
		if dE <= 0:
			spin_lattice_copy[i][j] *= -1
			# 3. If the new energy is less, keep the flipped value.
		ran_num = random.random()
		if dE > 0 and ran_num < P_flip:	#Acceptance probability
			spin_lattice_copy[i][j] *= -1
			# 4. If the new energy is more, only keep with probability P_flip

		#Addition of the net magnetization and magnetic susceptibility to its list
		#M_arr.append( np.mean(spin_lattice) )
		#M = np.mean(spin_lattice)
		#mag_sus_arr.append( mag_sus( spin_lattice ) )
		loop_counter += 1
		sweeps = loop_counter/(n**2)

		if abs( np.mean( spin_lattice_copy ) ) == 1:
			sweep_min = sweeps
			print 'Number of sweeps',sweeps,'at',1/beta,'K'
			sweeps = sweep_limit # ends the while loop

	print 'Moving onto next temperature'

	eqlb_M = np.mean( spin_lattice_copy )	#Equilibrium value for magnetization
	eqlb_Xv = mag_sus( spin_lattice_copy, beta )	#Equilibrium value for magnetic sus.

	#Stops the runtime timer
	stop_iterations_timer = timeit.default_timer()
	iterations_runtime = stop_iterations_timer - start_iterations_timer
	#print eqlb_M, eqlb_mag_sus

	return spin_lattice_copy, eqlb_M, eqlb_Xv, iterations_runtime, sweep_min

#------------------------------
#Results to output in terminal
#------------------------------

#The RHS of the equation below runs the iteration function for each value of temperature. It stores the results of each iteration in an array. All of these results arrays are stored in the LHS.
#results_array = [iteration(sweep_limit, spin_lattice_copy, beta) for sweep_limit, spin_lattice_copy, beta in zip(sweep_limits, spin_lattice_copies, beta_array)]
results_array = [iteration(spin_lattice_copy, beta) for spin_lattice_copy, beta in zip(spin_lattice_copies, beta_array)]

spin_lattices_array = [i[0] for i in results_array]
eqlb_M_values = [i[1] for i in results_array]
eqlb_Xv_values = [i[2] for i in results_array]
iterations_runtime_values = [i[3] for i in results_array]
sweep_min_values = [i[4] for i in results_array]

stop_program_timer = timeit.default_timer()
program_runtime = stop_program_timer - start_program_timer
print 'Runtime = ', program_runtime, 'seconds'
'''
#--------------------------------------------------------------------
Everything below is for plotting
#--------------------------------------------------------------------
'''
#------------------------------
#Plotting the statistical info
#------------------------------

plt.plot(temperatures, sweep_min_values)
plt.title('Number of Sweeps vs. Temperature')
plt.xlabel(r'Temperature, $K$')
plt.ylabel('Number of Sweeps')
plt.show()

'''


fig1, (ax2, ax3, ax4) = plt.subplots( ncols=3, figsize=(14, 7) ) #http://bit.ly/1mXPklK

#x3 = np.arange(0, temperatures, 1)
#x4 = np.arange(0, temperatures, 1)
#fit = 1 - np.exp(-x1/(25*n + n**2))

ax2.plot(temperatures, iterations_runtime_values, 'r')
ax2.set_title('Runtime')
ax2.set_xlabel(r'Temperature, $K$')
ax2.set_ylabel(r'Runtime, $t$ (s)')

ax3.plot(temperatures, eqlb_Xv_values, 'b')
ax3.set_title('Magnetic Susceptibility')
ax3.set_xlabel(r'Temperature, $K$')
ax3.set_ylabel(r'Magnetic Susceptibility, $\chi_\nu$')

ax4.plot(temperatures, eqlb_M_values, 'g')
ax4.set_title('Magnetisation')
ax4.set_xlabel(r'Temperature, $K$')
ax4.set_ylabel(r'Magnetisation, $M$ $(A/m^3)$')

fig1.subplots_adjust(left = 0.07, right = 0.96, wspace = 0.3)

'''

'''
#-----------------------------
#Plotting the spin lattices
#-----------------------------
'''
'''
#Figure with sub-plots
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(7,3.5)) #http://bit.ly/1mXPklK

#Move plots over to make room for the colour bar axis, which is defined there below
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.19, 0.02, 0.62])


#Generation of colour maps and then of the colour bar
ax1.imshow(original_spin_lattice, interpolation='nearest', cmap = colour_map, norm = norm)
ax1.set_title('Before')
img = ax2.imshow(spin_lattice, interpolation='nearest', cmap = colour_map, norm = norm)
ax2.set_title('After')
plt.colorbar(img, cax=cbar_ax, cmap=colour_map)

img = .imshow(spin_lattice, interpolation='nearest', cmap = colour_map, norm = norm)
ax2.set_title('After')

'''
'''
plt.ion()
#The colours I want my plots to be in. Grey for 1, and black for -1. Grey and black is easier on the eyes.
colour_map = mpl.colors.ListedColormap(['black', 'grey'])
bounds = [-1, 0, 1]
norm = mpl.colors.BoundaryNorm(bounds, colour_map.N)

for i in range( len(spin_lattices_array) ):
	plt.imshow(spin_lattices_array[i], interpolation='nearest', cmap = colour_map, norm = norm)
	plt.draw()
	print temperatures[i]
	time.sleep(0.001)
	plt.cla() #http://bit.ly/1P6zLEi

'''
'''
#--------------------------------------------------------------------
End of script
#--------------------------------------------------------------------
'''
