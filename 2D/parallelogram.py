#!/usr/bin/python
from dolfin import *
from matplotlib import pyplot as plt
import sys

import helper
import parameters
import neumann
import robin
import fundamental
import variance
import dirichlet
  
print
print "Parallelogram"            


mesh_name = "parallelogram"
mesh_obj = helper.make_2D_parallelogram( 100, 100, 1.3 )
kappa = 11. # Killing rate
num_samples = 0

container = parameters.Container( mesh_name,
                                  mesh_obj,
                                  kappa,
                                  2, # power = 2 in all my simulations
                                  num_samples )

mode = "color"

# print "neumann variance"
variance.neumann_variance( container, mode )

# print "robin variance"
# variance.robin_variance( container, mode )

print "neumann"
neumann.neumann        ( container, mode )
    
print "fundamental"
fundamental.fundamental( container, mode )

# print "mixed"
# robin.mixed            ( container, mode )

print "improper"
robin.improper         ( container, mode )

print "naive"
robin.naive            ( container, mode )

# print "dirichlet"
# dirichlet.dirichlet    ( container, mode )

plt.legend()
plt.savefig( "../../PriorCov/" + mesh_name +"_section.png" )
plt.close()

print
