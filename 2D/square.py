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

mesh_obj = helper.refine( mesh_name )

container = parameters.Container( mesh_name,
                                  mesh_obj,
                                  kappa,
                                  1.0 )

print "neumann"
neumann.neumann        ( container, mode )

print "fundamental"
fundamental.fundamental( container, mode )
    
print "dirichlet"
dirichlet.dirichlet    ( container, mode )
    

