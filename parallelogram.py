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

mesh_obj = helper.refine( "parallelogram",
                          nor = 1,
                          tol = 0.35,
                          factor = 0.66 )

container = container.Container( "parallelogram",
                                 mesh_obj,
                                 5. ) # == kappa == Killing rate

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

print "neumann variance"
start_time = time.time()
variance.variance( container, "neumann" )
print "Run time: " + str( time.time() - start_time )
print

print "dirichlet"
start_time = time.time()
regular.ordinary(container, "dirichlet" )
print "Run time: " + str( time.time() - start_time )
print

print "naive" 
start_time = time.time()
regular.ordinary(container, "naive robin" )
print "Run time: " + str( time.time() - start_time )
print

print "naive robin variance"
start_time = time.time()
variance.variance( container, "naive robin" )
print "Run time: " + str( time.time() - start_time )
print

print "improper"
start_time = time.time()
regular.ordinary(container, "improper robin" )
print "Run time: " + str( time.time() - start_time )
print

print "improper robin variance"
start_time = time.time()
variance.variance( container, "improper robin" )
print "Run time: " + str( time.time() - start_time )
print

print "mixed"
start_time = time.time()
regular.ordinary(container, "mixed robin" )
print "Run time: " + str( time.time() - start_time )
print

print "mixed robin variance"
start_time = time.time()
variance.variance( container, "mixed robin" )
print "Run time: " + str( time.time() - start_time )
print
