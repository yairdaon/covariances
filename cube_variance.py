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

mesh_obj = helper.refine_cube( 25, 67, 53, nor = 0 )
                               # nor = 0, 
                               # tol = 0.15,
                               # factor = .8,
                               # show = False,
                               # refine_source = True,
                               # refine_face = False,
                               # refine_cross = True )

container = container.Container( "cube",
                                 mesh_obj,
                                 5. ) # == kappa == Killing rate
                                 

print "fundamental"
start_time = time.time()
fundamental3D.fundamental( container )
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
