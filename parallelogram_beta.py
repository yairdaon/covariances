#!/usr/bin/python
import numpy as np
import math

from dolfin import *

import container
import helper
import betas
from helper import dic as dic

print
print "Parallelogram beta"            

container = container.Container( "parallelogram",
                                 dic["parallelogram"](), # get the mesh, lazily
                                 dic["parallelogram"].alpha,
                                 gamma = 1 )

#plot( container.mesh_obj, interactive= True )
    
beta = betas.Beta2DAdaptive( container )

A = dic["parallelogram"].transformation

# Return coordiantes along the direction Ae_2
xy = lambda s: ( s*A[0,1] , s*A[1,1] )

# Outward pointing normal to the edge Ae_2
nn = np.array( [ -A[1,1], A[0,1] ] ) / math.sqrt( A[0,1]**2 + A[1,1]**2 )

pt_list = np.linspace(0,1,102)
beta_file = "../PriorCov/data/parallelogram/beta.txt"
helper.empty_file( beta_file )

for s in pt_list:
    helper.add_point( beta_file, s, np.dot( beta(xy(s)) , nn ) )
