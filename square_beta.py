#!/usr/bin/python
import numpy as np
import math
import time

from dolfin import *

import container
import helper
import betas2D
                 
mesh_obj = UnitSquareMesh( 673, 673 )

container = container.Container( "square",
                                 mesh_obj,
                                 5.0 ) # == kappa == Killing rate
                                 

imp_beta = betas2D.Beta( container, "imp" )
mix_beta = betas2D.Beta( container, "mix" )

x = lambda s: 0.0
y = lambda s: s

pt_list = np.linspace(0.00,1.0,101)
imp_beta_file = "data/square/imp_beta.txt"
mix_beta_file = "data/square/mix_beta.txt"
helper.empty_file( imp_beta_file, mix_beta_file )

imp_list = []
mix_list = []
for s in pt_list:
    imp = -imp_beta( x(s), y(s) )[0]
    helper.add_point( imp_beta_file, s, imp )
    imp_list.append( imp )
    mix = -mix_beta( x(s), y(s) )[0]
    helper.add_point( mix_beta_file, s, mix )
    mix_list.append( mix )
