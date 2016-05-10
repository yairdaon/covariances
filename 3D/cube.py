#!/usr/bin/python
from dolfin import *
from matplotlib import pyplot as plt
import sys
import time

import container
import helper
import fundamental
import variance
import regular

print
print "Cube"            


mesh_name = "cube"
mesh_obj = helper.refine_cube( 18, 18, 18, 
                               nor = 0, 
                               tol = 0.2,
                               factor = 0.4,
                               show = True )

container = container.Container( mesh_name,
                                 mesh_obj,
                                 11. # == kappa == Killing rate
                             )

mode = "color"

print "fundamental"
start_time = time.time()
fundamental.fundamental( container, mode )
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
