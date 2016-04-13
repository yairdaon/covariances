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

mesh_name = "square"
mode = "color"              
kappa = 11. # Killing rate
num_samples = 0

mesh_obj = UnitSquareMesh( 100, 100 )

container = parameters.Container( mesh_name,
                                  mesh_obj,
                                  kappa,
                                  2, # power = 2 in all my simulations
                                  num_samples )
print "neumann"
neumann.neumann        ( container, mode )

print "fundamental"
fundamental.fundamental( container, mode )
    
print "dirichlet"
dirichlet.dirichlet    ( container, mode )
    

