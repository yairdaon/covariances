#!/usr/bin/python
import time

import helper
import container
import fundamental2D
import variance
import regular
from helper import dic as dic

container = container.Container( "square",
                                 dic["square"](), 
                                 dic["square"].alpha,
                                 gamma = 1 )

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

print "dirichlet"
start_time = time.time()
regular.ordinary(container, "dirichlet" )
print "Run time: " + str( time.time() - start_time )
print
