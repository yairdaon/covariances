#!/usr/bin/python
import scipy.integrate as spi
import scipy as sp
import numpy as np
import math
import sys
import time
import pdb
import os
import matplotlib.pyplot as plt

from dolfin import *

import container
import helper
import betas2D
   
kappa = 5.0
                  
mesh_name = "square"
mesh_obj = UnitSquareMesh( 673, 673 )


container = container.Container( mesh_name,
                                 mesh_obj,
                                 kappa, # == kappa == Killing rate
                                 num_samples = 0 )

imp_beta = betas2D.Beta( container, "imp" )
mix_beta = betas2D.Beta( container, "mix" )

x = lambda s: 0.0
y = lambda s: s

pt_list = np.linspace(0.05,0.95,87)
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
