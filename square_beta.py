#!/usr/bin/python
import numpy as np
import math
import time

from dolfin import *

import container
import helper
import betas2D
from helper import dic as dic

container = container.Container( "square",
                                 dic["square"](), # get the mesh, lazily
                                 dic["square"].kappa ) # == kappa == Killing rate


imp_beta = betas2D.Beta( container, "imp" )
mix_beta = betas2D.Beta( container, "mix" )

xy = lambda s: ( 0.0, s )

pt_list = np.linspace(0.00,1.0,101)
imp_beta_file = "../PriorCov/data/square/imp_beta.txt"
mix_beta_file = "../PriorCov/data/square/mix_beta.txt"
helper.empty_file( imp_beta_file, mix_beta_file )

for s in pt_list:
    imp = -imp_beta( xy(s) )[0]
    helper.add_point( imp_beta_file, s, imp )

    mix = -mix_beta( xy(s) )[0]
    helper.add_point( mix_beta_file, s, mix )

