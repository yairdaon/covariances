#!/usr/bin/python
from dolfin import *
import time

import helper
import container
import fundamental2D
import variance
import regular


mesh_name = "square"
mode = "color"              
kappa = 11. # Killing rate

mesh_obj = helper.refine( mesh_name,
                          nor = 4,
                          tol = 0.15,
                          factor = 0.5 )

container = container.Container( mesh_name,
                                  mesh_obj,
                                  kappa )

print "fundamental"
start_time = time.time()
fundamental2D.fundamental( container, mode )
print "Run time: " + str( time.time() - start_time )
print

print "neumann"
start_time = time.time()
regular.ordinary(container, mode, "neumann" )
print "Run time: " + str( time.time() - start_time )
print

print "dirichlet"
start_time = time.time()
regular.ordinary(container, mode, "dirichlet" )
print "Run time: " + str( time.time() - start_time )
print
