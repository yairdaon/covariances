#!/usr/bin/python
from dolfin import *
from matplotlib import pyplot as plt
import sys

import helper
import parameters
import neumann
import robin
import fundamental
import variance
import integrated
import dirichlet

if len( sys.argv ) > 1:
    mesh_name = sys.argv[1]
else:
    mesh_name = "square"
              
kappa = 11. # Killing rate
num_samples = 0
nx = 500
ny = 500
nz = 10

if mesh_name == "square":
    mesh_obj = UnitSquareMesh( 5, 5 )
    dim = 2
elif mesh_name == "parallelogram":
    mesh_obj = helper.make_2D_parallelogram( nx, ny, 1.3 )
    dim = 2
elif mesh_name == "cube":
    mesh_obj = UnitCubeMesh( nx, ny, nz )
    dim = 3
else:
    mesh_obj = Mesh( "meshes/" + mesh_name + ".xml" )
    dim = 2

if dim == 2:
    mode = "color"
else:
    mode = "auto" 

refines = 0
if len( sys.argv ) == 3:
    refines = int( sys.argv[2] )
    for i in range( refines ):
        mesh_obj = refine( mesh_obj )
        
    mesh_name = mesh_name + "_" + str(refines)

container = parameters.Container( mesh_name,
                                  mesh_obj,
                                  kappa,
                                  2, # power = 2 in all my simulations
                                  num_samples )


if len( sys.argv ) == 2:

    print "neumann variance"
    variance.neumann_variance( container, mode )
    
    print "robin variance"
    variance.robin_variance( container, mode )

    print "neumann"
    neumann.neumann        ( container, mode )
    
    print "fundamental"
    fundamental.fundamental( container, mode )
    
    print "mixed"
    robin.mixed            ( container, mode )
    
    print "improper"
    robin.improper         ( container, mode )
    
    print "naive"
    robin.naive            ( container, mode )
    
    print "dirichlet"
    dirichlet.dirichlet    ( container, mode )
    
elif len( sys.argv ) == 3:
    
    print "improper refined " + str( refines )
    robin.improper         ( container, mode )

    print "robin variance refined " + str( refines )
    variance.robin_variance( container, mode )

if mesh_name == "square" or "cube":
    plt.legend()
    plt.savefig( "../../PriorCov/" + mesh_name +"_section.png" )
    plt.close()

if mode == "auto":
    interactive()

