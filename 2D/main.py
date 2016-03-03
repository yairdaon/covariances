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

mesh_name = sys.argv[1]

if mesh_name == "square":
    mesh_obj = UnitSquareMesh( 100, 40 )
else:
    mesh_obj = Mesh( "meshes/" + mesh_name + ".xml" )
   

kappa = 11. # Killing rate
dim = 2 # dimension
nu = 1
num_samples = 50000

mode = "color"
# mode = "auto" # makes it 3D!!!

container = parameters.Container( mesh_name,
                                  mesh_obj,
                                  kappa,
                                  dim,
                                  nu,
                                  num_samples )

fundamental.fundamental( container, mode )
robin.improper( container, mode )
neumann.neumann( container, mode )
variance.variance( container, mode )
#robin.robin( container, mode )

if mesh_name == "square":
    plt.legend()
    plt.savefig( "../../PriorCov/1D.png" )
    plt.close()

if mode == "auto":
    interactive()

