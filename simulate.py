#!/usr/bin/python
import time
import sys

import helper
import container
import fundamental
import variance
import regular
from helper import dic as dic
  
mesh_name = sys.argv[1]
print
print mesh_name


container = container.Container( mesh_name,
                                 dic[mesh_name](), # get the mesh
                                 dic[mesh_name].alpha,
                                 gamma = 1,
                                 quad = "std" )


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
fundamental.fundamental( container )
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
