#!/usr/bin/python
from dolfin import *
import math

import helper
import parameters
import neumann
import robin
import fundamental
import variance
import dirichlet

print
print "Green's functions and varainces"              

mesh_name = "antarctica1"
#mesh_name = "dolfin_fine"
mesh_obj = helper.refine( mesh_name, show = True )

delta = 1e-5
kappa = math.sqrt( delta )
gamma = 1.0

container = parameters.Container( mesh_name,
                                  mesh_obj,
                                  kappa,
                                  gamma )
mode = "color"

# print "fundamental"
# fundamental.fundamental( container, mode )

print "neumann"
neumann.neumann        ( container, mode )

# print "naive"
# robin.naive            ( container, mode )

# print "dirichlet"
# dirichlet.dirichlet    ( container, mode )

# print "mixed"
# robin.mixed            ( container, mode )

# print "improper"
# robin.improper         ( container, mode )

# print "neumann variance"
# variance.neumann_variance( container, mode )

# print "naive robin variance"
# variance.naive_robin_variance( container, mode )

# print "mixed robin variance"
# variance.mixed_robin_variance( container, mode )

# print "improper robin variance"
# variance.improper_robin_variance( container, mode )

print
