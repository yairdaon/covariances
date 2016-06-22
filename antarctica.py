#!/usr/bin/python
from dolfin import *
import math
import time

import helper
import container
import fundamental2D
import variance
import regular

print
print "Green's functions and variances"              

mesh_name = "antarctica3"

mesh_obj = helper.refine( mesh_name, 
                          nor = 2,
                          tol = 200.,
                          factor = 0.6,
                          show = False )

delta = 1e-5
kappa = math.sqrt( delta )
gamma = 1.

container = container.Container( mesh_name,
                                 mesh_obj,
                                 kappa, # == Killing rate
                                 gamma = gamma ) # prefactor of laplacian


container.mesh_name = "antarctica"

# print "fundamental"
# start_time = time.time()
# fundamental2D.fundamental( container )
# print "Run time: " + str( time.time() - start_time )
# print

# print "neumann"
# start_time = time.time()
# regular.ordinary(container, "neumann" )
# print "Run time: " + str( time.time() - start_time )
# print

# print "neumann variance"
# start_time = time.time()
# variance.variance( container, "neumann" )
# print "Run time: " + str( time.time() - start_time )
# print

# print "dirichlet"
# start_time = time.time()
# regular.ordinary(container,  "dirichlet" )
# print "Run time: " + str( time.time() - start_time )
# print

# print "dirichlet variance"
# start_time = time.time()
# variance.variance( container, "dirichlet" )
# print "Run time: " + str( time.time() - start_time )
# print

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
