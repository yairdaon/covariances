#!/usr/bin/python
import numpy as np
import math
import time

from dolfin import *

import container
import helper
import betas
from helper import dic as dic

container = container.Container( "square",
                                 dic["square"](),
                                 dic["square"].alpha,
                                 gamma = 1 ) 

beta = betas.Beta2DAdaptive( container )

xy = lambda s: ( 0.0, s )

pt_list = np.linspace(0.00,1.0,107)

beta_file = "../PriorCov/data/square/beta.txt"
helper.empty_file( beta_file )

for s in pt_list:
    helper.add_point( beta_file, s, -beta( xy(s) )[0] )

