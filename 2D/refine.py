#!/usr/bin/python
from dolfin import *
import sys

import helper
import parameters
import neumann
import robin
import fundamental
import variance
import dirichlet
              
print
print "Mesh refinements"

kappa = 11. # Killing rate
num_samples = 2

mesh_obj = Mesh( "meshes/dolfin_fine.xml" )
  
mode = "color"

for i in range( int( sys.argv[1] ) ):
    mesh_name = "dolfin_fine_" + str(i)

    container = parameters.Container( mesh_name,
                                      mesh_obj,
                                      kappa,
                                      2, # power = 2 in all my simulations
                                      num_samples )

    print "robin variance"
    variance.robin_variance( container, mode )
    
    print "mixed"
    robin.mixed            ( container, mode )
    
    print "improper"
    robin.improper         ( container, mode )
    
    mesh_obj = refine( mesh_obj )

print
