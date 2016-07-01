#!/usr/bin/python
from dolfin import *

import helper
import container
import fundamental2D
import variance
import regular
from helper import dic as dic

container = container.Container( "square",
                                 dic["square"](), # get the mesh, lazily
                                 dic["square"].kappa ) # == kappa == Killing rate

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

print "dirichlet"
start_time = time()
regular.ordinary(container, "dirichlet" )
print "Run time: " + str( time() - start_time )
print
