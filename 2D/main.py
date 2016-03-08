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
    dim = 2
elif mesh_name == "cube":
    mesh_obj = UnitCubeMesh( 100, 40, 40 )
    dim = 3
else:
    mesh_obj = Mesh( "meshes/" + mesh_name + ".xml" )
    dim = 2

kappa = 11. # Killing rate
nu = 2 - dim / 2.0
num_samples = 50000

if dim == 2:
    mode = "color"
else:
    mode = "auto" 

container = parameters.Container( mesh_name,
                                  mesh_obj,
                                  kappa,
                                  dim,
                                  nu,
                                  num_samples )
variance.neumann_variance( container, mode )
variance.robin_variance( container, mode )
fundamental.fundamental( container, mode )
neumann.neumann( container, mode )
if dim == 2:
    robin.improper( container, mode )
#robin.robin( container, mode )

if mesh_name == "square" or "cube":
    plt.legend()
    plt.savefig( "../../PriorCov/" + mesh_name +"_section.png" )
    plt.close()

if mode == "auto":
    interactive()

