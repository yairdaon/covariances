#!/usr/bin/python
import numpy as np
import math

from dolfin import *

import container
import helper
import betas
from helper import dic as dic

print
print "Cube beta"            

container = container.Container( "cube",
                                 dic["cube"](), # get the mesh, lazily
                                 dic["cube"].alpha,
                                 gamma = 1 )
    
beta = betas.BetaCubeAdaptive( container )

xyz = lambda s: ( 0.0, 0.5, s )

pt_list = np.linspace(0.0,1.0,106)
file_name = "../PriorCov/data/cube/beta.txt"
helper.empty_file( file_name )    
for s in pt_list:
    helper.add_point( file_name, s, -beta( xyz(s) )[0] )
