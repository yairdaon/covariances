#!/usr/bin/python
from dolfin import *
import sys
import time
import pdb

import container
import helper
import fundamental3D
import variance
import regular

print
print "Cube"            

mesh_name = "cube"
mesh_obj = helper.refine_cube( 6, 6, 6, 
                               nor = 4, 
                               tol = 0.2,
                               factor = 1.0,
                               # show = True,
                               greens = True )

container = container.Container( mesh_name,
                                 mesh_obj,
                                 5., # == kappa == Killing rate
                                 num_samples = 0 )

mode = "color"

# print "fundamental"
# start_time = time.time()
# fundamental3D.fundamental( container, mode )
# print "Run time: " + str( time.time() - start_time )
# print

# print "mixed"
# start_time = time.time()
# regular.ordinary(container, mode, "mixed robin" )
# print "Run time: " + str( time.time() - start_time )
# print

# print "mixed robin variance"
# start_time = time.time()
# variance.variance( container, mode, "mixed robin" )
# print "Run time: " + str( time.time() - start_time )
# print

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

