#!/usr/bin/python
from dolfin import *
import math
import time

import helper
import container
import neumann
import robin
import fundamental
import variance
import dirichlet

print
print "Green's functions and variances"              

mesh_name = "antarctica1"

mesh_obj = helper.refine( mesh_name, nor = 2, show = False )

delta = 1e-5 
kappa = math.sqrt( delta )
gamma = 10.0 

container = container.Container( mesh_name,
                                 mesh_obj,
                                 kappa,
                                 gamma )
mode = "color"

print "fundamental"
start_time = time.time()
fundamental.fundamental( container, mode )
print "Run time: " + str( time.time() - start_time )
print

print "dirichlet"
start_time = time.time()
dirichlet.dirichlet    ( container, mode )
print "Run time: " + str( time.time() - start_time )
print

print "neumann"
start_time = time.time()
neumann.neumann        ( container, mode )
print "Run time: " + str( time.time() - start_time )
print

print "neumann variance"
start_time = time.time()
variance.neumann_variance( container, mode )
print "Run time: " + str( time.time() - start_time )
print

print "naive" 
start_time = time.time()
robin.naive            ( container, mode )
print "Run time: " + str( time.time() - start_time )
print

print "naive robin variance"
start_time = time.time()
variance.naive_robin_variance( container, mode )
print "Run time: " + str( time.time() - start_time )
print

print "mixed"
start_time = time.time()
robin.mixed            ( container, mode )
print "Run time: " + str( time.time() - start_time )
print

print "mixed robin variance"
start_time = time.time()
variance.mixed_robin_variance( container, mode )
print "Run time: " + str( time.time() - start_time )
print

print "improper"
start_time = time.time()
robin.improper         ( container, mode )
print "Run time: " + str( time.time() - start_time )
print

print "improper robin variance"
start_time = time.time()
variance.improper_robin_variance( container, mode )
print "Run time: " + str( time.time() - start_time )
print
