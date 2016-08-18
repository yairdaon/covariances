#!/usr/bin/python
from dolfin import *
import os

import helper
import container
import fundamental2D
import variance
import regular
from helper import dic as dic

os.system( "rm -rvf data/parallelogram/*" )



cot = container.Container( "parallelogram",
                           dic["parallelogram"](),
                           dic["parallelogram"].kappa ) # == kappa == Killing rate

print "neumann"
start_time = time()
regular.ordinary( cot, "neumann refined" )
print "Run time: " + str( time() - start_time )
print

print "naive"
start_time = time()
regular.ordinary( cot, "naive refined" )
print "Run time: " + str( time() - start_time )
print

print "dirichlet"
start_time = time()
regular.ordinary( cot, "dirichlet refined" )
print "Run time: " + str( time() - start_time )
print

for i in range( 8, 25 ):

    n = int(1.5**i)
    print "n = " + str(n)
    dic["parallelogram"].x = n
    dic["parallelogram"].y = n

    cot = container.Container( "parallelogram",
                               helper.get_mesh( "parallelogram" ),
                               dic["parallelogram"].kappa ) # == kappa == Killing rate
    
    print "neumann"
    start_time = time()
    regular.ordinary( cot, "neumann " + str(n) + " X " + str(n) )
    print "Run time: " + str( time() - start_time )
    print

    print "naive"
    start_time = time()
    regular.ordinary( cot, "naive " + str(n) + " X " + str(n) )
    print "Run time: " + str( time() - start_time )
    print

    print "dirichlet"
    start_time = time()
    regular.ordinary( cot, "dirichlet " + str(n) + " X " + str(n) )
    print "Run time: " + str( time() - start_time )
    print



