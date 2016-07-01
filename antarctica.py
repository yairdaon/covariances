#!/usr/bin/python
import math

from dolfin import *

import helper
import container
import fundamental2D
import variance
import regular
from helper import dic as dic

print
print "Green's functions and variances"              

container = container.Container( "antarctica",
                                 dic["antarctica"](), # get the mesh, lazily
                                 dic["antarctica"].kappa, # == Killing rate
                                 gamma = dic["antarctica"].gamma ) # prefactor of laplacian

print "fundamental"
start_time = time()
fundamental2D.fundamental( container )
print "Run time: " + str( time() - start_time )
print

print "neumann"
start_time = time()
regular.ordinary(container, "neumann" )
print "Run time: " + str( time() - start_time )
print

print "neumann variance"
start_time = time()
variance.variance( container, "neumann" )
print "Run time: " + str( time() - start_time )
print

print "dirichlet"
start_time = time()
regular.ordinary(container,  "dirichlet" )
print "Run time: " + str( time() - start_time )
print

print "dirichlet variance"
start_time = time()
variance.variance( container, "dirichlet" )
print "Run time: " + str( time() - start_time )
print

print "naive" 
start_time = time()
regular.ordinary(container, "naive robin" )
print "Run time: " + str( time() - start_time )
print

print "naive robin variance"
start_time = time()
variance.variance( container, "naive robin" )
print "Run time: " + str( time() - start_time )
print

print "improper"
start_time = time()
regular.ordinary(container, "improper robin" )
print "Run time: " + str( time() - start_time )
print

print "improper robin variance"
start_time = time()
variance.variance( container, "improper robin" )
print "Run time: " + str( time() - start_time )
print

print "mixed"
start_time = time()
regular.ordinary(container, "mixed robin" )
print "Run time: " + str( time() - start_time )
print

print "mixed robin variance"
start_time = time()
variance.variance( container, "mixed robin" )
print "Run time: " + str( time() - start_time )
print
