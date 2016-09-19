#!/usr/bin/python
import time

import helper
import container
import fundamental2D
import variance
import regular
from helper import dic as dic
  
print
print "Parallelogram"            

container = container.Container( "parallelogram",
                                 dic["parallelogram"](), # get the mesh
                                 dic["parallelogram"].alpha,
                                 gamma = 1,
                                 quad = "radial" )


print "roininen" 
start_time = time.time()
regular.ordinary(container, "roininen robin" )
print "Run time: " + str( time.time() - start_time )
print

print "ours"
start_time = time.time()
regular.ordinary(container, "ours" )
print "Run time: " + str( time.time() - start_time )
print

print "ours robin variance"
start_time = time.time()
variance.variance( container, "ours" )
print "Run time: " + str( time.time() - start_time )
print

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
