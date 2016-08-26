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
                                 dic["antarctica"].alpha, # == Killing rate
                                 gamma = dic["antarctica"].gamma,
                                 radial = False ) 

print "roininen" 
start_time = time()
regular.ordinary(container, "roininen" )
print "Run time: " + str( time() - start_time )
print

print "ours"
start_time = time()
regular.ordinary(container, "ours" )
print "Run time: " + str( time() - start_time )
print

print "naive robin variance"
start_time = time()
variance.variance( container, "roininen" )
print "Run time: " + str( time() - start_time )
print

print "mixed robin variance"
start_time = time()
variance.variance( container, "ours" )
print "Run time: " + str( time() - start_time )
print

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

