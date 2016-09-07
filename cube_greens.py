#!/usr/bin/python
from dolfin import *

import container
import helper
import fundamental3D
import variance
import regular

from helper import dic as dic
print
print "Cube"            


cot = container.Container( "cube",
                           dic["cube"](),
                           dic["cube"].alpha,
                           quad = "std" ) 

print "fundamental"
start_time = time()
fundamental3D.fundamental( cot )
print "Run time: " + str( time() - start_time )
print

print "ours"
start_time = time()
regular.ordinary( cot, "ours" )
print "Run time: " + str( time() - start_time )
print

print "ours variance"
start_time = time()
variance.variance( cot, "ours" )
print "Run time: " + str( time() - start_time )
print

print "neumann"
start_time = time()
regular.ordinary( cot, "neumann" )
print "Run time: " + str( time() - start_time )
print

print "neumann variance"
start_time = time()
variance.variance( cot, "neumann" )
print "Run time: " + str( time() - start_time )
print

print "dirichlet"
start_time = time()
regular.ordinary( cot, "dirichlet" )
print "Run time: " + str( time() - start_time )
print

print "Roninen" 
start_time = time()
regular.ordinary( cot, "roininen" )
print "Run time: " + str( time() - start_time )
print

print "Roininen robin variance"
start_time = time()
variance.variance( cot, "roininen" )
print "Run time: " + str( time() - start_time )
print

