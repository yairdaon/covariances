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
print "Cube"            


mesh_name = "cube"
mesh_obj = helper.refine_cube( 20, 20, 20, nor = 2 )

kappa = 11. # Killing rate
num_samples = 0

container = parameters.Container( mesh_name,
                                  mesh_obj,
                                  kappa,
                                  2, # power = 2 in all my simulations
                                  num_samples )

mode = "color"

print "fundamental"
fundamental.fundamental( container, mode )

print "neumann"
neumann.neumann        ( container, mode )

print "naive"
robin.naive            ( container, mode )

print "dirichlet"
dirichlet.dirichlet    ( container, mode )

print "mixed"
robin.mixed            ( container, mode )

print "neumann variance"
variance.neumann_variance( container, mode )

print "naive robin variance"
variance.naive_robin_variance( container, mode )

print "mixed robin variance"
variance.mixed_robin_variance( container, mode )

print "Done."
