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

cot = container.Container( "antarctica",
                           dic["antarctica"](), # get the mesh, lazily
                           dic["antarctica"].alpha, # == Killing rate
                           gamma = dic["antarctica"].gamma,
                           quad = "std" ) 

print "roininen" 
start_time = time()
regular.ordinary(cot, "roininen" )
print "Run time: " + str( time() - start_time )
print

print "ours"
start_time = time()
regular.ordinary(cot, "ours" )
print "Run time: " + str( time() - start_time )
print

print "roininen robin variance"
start_time = time()
variance.variance( cot, "roininen" )
print "Run time: " + str( time() - start_time )
print

print "ours variance"
start_time = time()
variance.variance( cot, "ours" )
print "Run time: " + str( time() - start_time )
print

assert False
print "fundamental"
start_time = time()
fundamental2D.fundamental( cot )
print "Run time: " + str( time() - start_time )
print

print "neumann"
start_time = time()
regular.ordinary(cot, "neumann" )
print "Run time: " + str( time() - start_time )
print

print "neumann variance"
start_time = time()
variance.variance( cot, "neumann" )
print "Run time: " + str( time() - start_time )
print

print "dirichlet"
start_time = time()
regular.ordinary(cot,  "dirichlet" )
print "Run time: " + str( time() - start_time )
print

print "dirichlet variance"
start_time = time()
variance.variance( cot, "dirichlet" )
print "Run time: " + str( time() - start_time )
print

