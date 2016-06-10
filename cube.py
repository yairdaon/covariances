#!/usr/bin/python
'''
This script creates all the pvd and vtu files that are 
used in the paper for 3D visualizations of our methods on
the unit cube. 
'''

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

# Refine the cube around the source of
# Green's function.
mesh_obj = helper.refine_cube( 7, 7, 7, 
                               nor = 4, 
                               tol = 0.2,
                               factor = 1.,
                               show = False,
                               greens = True,
                               variance = False )

# Contains all parameters data etc. required
# to use our methods
container = container.Container( mesh_name,
                                 mesh_obj,
                                 11., # == kappa == Killing rate
                                 num_samples = 0 )

mode = "color"

print "fundamental"
start_time = time.time()
fundamental3D.fundamental( container, mode )
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
