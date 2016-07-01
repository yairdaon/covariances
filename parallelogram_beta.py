#!/usr/bin/python
import numpy as np
import math

from dolfin import *

import container
import helper
import betas2D
from helper import dic as dic

print
print "Parallelogram beta"            

container = container.Container( "parallelogram",
                                 dic["parallelogram"](), # get the mesh, lazily
                                 dic["parallelogram"].kappa )# == kappa == Killing rate
                             
imp_beta = betas2D.Beta( container, "imp" )
mix_beta = betas2D.Beta( container, "mix" )

A = dic["parallelogram"].transformation

# Return coordiantes along the direction Ae_2
xy = lambda s: ( s*A[0,1] , s*A[1,1] )

# Outward pointing normal to the edge Ae_2
nn = np.array( [ -A[1,1], A[0,1] ] ) / math.sqrt( A[0,1]**2 + A[1,1]**2 )

pt_list = np.linspace(0,1,102)
imp_beta_file = "../PriorCov/data/parallelogram/imp_beta.txt"
mix_beta_file = "../PriorCov/data/parallelogram/mix_beta.txt"
helper.empty_file( imp_beta_file, mix_beta_file )

for s in pt_list:
   
    imp = np.dot( imp_beta(xy(s)) , nn )
    helper.add_point( imp_beta_file, s, imp )
   
    mix = np.dot( mix_beta(xy(s)) , nn )
    helper.add_point( mix_beta_file, s, mix )
