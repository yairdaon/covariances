#!/usr/bin/python
from dolfin import *

import helper
import parameters
import neumann
import robin
import fundamental
import variance
import dirichlet

print
print "Green's functions and varainces"              

mesh_name = "dolfin_coarse"
mesh_obj = Mesh( "meshes/" + mesh_name + ".xml" )
kappa = 11. # Killing rate
num_samples = 0


container = parameters.Container( mesh_name,
                                  mesh_obj,
                                  kappa,
                                  2, # power = 2 in all my simulations
                                  num_samples )

mode = "color"

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

print
