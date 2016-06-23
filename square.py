#!/usr/bin/python
from dolfin import *
import time

import helper
import container
import fundamental2D
import variance
import regular

mesh_obj = helper.refine( mesh_name,
                          nor = 4,
                          tol = 0.15,
                          factor = 0.5 )

container = container.Container( "square", # mesh_name
                                 mesh_obj,
                                 5. ) # Killing rate

print "fundamental"
start_time = time.time()
fundamental2D.fundamental( container )
print "Run time: " + str( time.time() - start_time )
print

print "neumann"
start_time = time.time()
regular.ordinary(container, "neumann" )
print "Run time: " + str( time.time() - start_time )
print

print "dirichlet"
start_time = time.time()
regular.ordinary(container, "dirichlet" )
print "Run time: " + str( time.time() - start_time )
print
