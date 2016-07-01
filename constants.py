import numpy as np
import math

import helper

# The declaration ... = lambda: None is only there
# to create an empty object, since lambda fuctions
# are objects in python is this is super clean IMO.

dic = {}

dic["square"] = lambda: helper.get_mesh( "square" )
dic["square"].kappa = 11.0
dic["square"].x = 555
dic["square"].y = 555
dic["square"].source = np.array( [ 0.05    , 0.5   ] ) 



dic["parallelogram"] = lambda: helper.get_refined_mesh( "parallelogram", nor=3, tol=0.5, factor=0.55, greens=True )
dic["parallelogram"].kappa = 11.0
dic["parallelogram"].x = 100
dic["parallelogram"].y = 100
dic["parallelogram"].s = 1.0
dic["parallelogram"].transformation = np.array( [ [ 2.5 , 1.  ],
                                                  [ 1.  , 2.5 ] ] )
dic["parallelogram"].source = np.array( [ 0.025   , 0.025 ] ) 



dic["antarctica"] = lambda: helper.get_refined_mesh( "antarctica", nor=0 )
dic["antarctica"].source        = np.array( [ 7e2     , 5e2   ] )
dic["antarctica"].center_source = np.array( [ -1.5e3  , 600.0 ] ) 
dic["antarctica"].delta = 1e-5
dic["antarctica"].kappa = math.sqrt( dic["antarctica"].delta )
dic["antarctica"].gamma = 1.



dic["cube"] = lambda: None
dic["cube"].kappa = 5.
n = 7#2
dic["cube"].x = n
dic["cube"].y = n
dic["cube"].z = n 
dic["cube"].source    = np.array( [ 0.05  ,  0.5,  0.5] ) 
dic["cube"].equations = np.array( [ [1    ,    1,    1],
                                    [0.125, 0.25,  0.5], 
                                    [0.75 ,    1,    1]
                                    ] )
dic["cube"].slope = 0.3
dic["cube"].coefficients = np.array( [1, 0.5, dic["cube"].slope ] )
dic["cube"].a, dic["cube"].b, dic["cube"].c = np.linalg.solve( dic["cube"].equations, dic["cube"].coefficients )
dic["cube"].s  = 1.

dic["dolfin"] = lambda: None
dic["dolfin"].source = np.array( [ 0.45    , 0.65  ] ) 
