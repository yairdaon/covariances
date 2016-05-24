#!/usr/bin/python
from dolfin import *
import time

import helper
import container
import fundamental2D
import variance
import regular
  
print
print "Parallelogram"            

mesh_name = "parallelogram"
mesh_obj = helper.refine( mesh_name,
                          nor = 3,
                          tol = 0.35,
                          factor = 0.66 )

container = container.Container( mesh_name,
                                 mesh_obj,
                                 11. # == kappa == Killing rate
                             )
mode = "color"

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

print "neumann variance"
start_time = time.time()
variance.variance( container, mode, "neumann" )
print "Run time: " + str( time.time() - start_time )
print

print "dirichlet"
start_time = time.time()
regular.ordinary(container, mode, "dirichlet" )
print "Run time: " + str( time.time() - start_time )
print

print "naive" 
start_time = time.time()
regular.ordinary(container, mode, "naive robin" )
print "Run time: " + str( time.time() - start_time )
print

print "naive robin variance"
start_time = time.time()
variance.variance( container, mode, "naive robin" )
print "Run time: " + str( time.time() - start_time )
print

print "improper"
start_time = time.time()
regular.ordinary(container, mode, "improper robin" )
print "Run time: " + str( time.time() - start_time )
print

print "improper robin variance"
start_time = time.time()
variance.variance( container, mode, "improper robin" )
print "Run time: " + str( time.time() - start_time )
print

print "mixed"
start_time = time.time()
regular.ordinary(container, mode, "mixed robin" )
print "Run time: " + str( time.time() - start_time )
print

print "mixed robin variance"
start_time = time.time()
variance.variance( container, mode, "mixed robin" )
print "Run time: " + str( time.time() - start_time )
print
