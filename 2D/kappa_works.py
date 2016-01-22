#!/usr/bin/python
from dolfin import *
import numpy as np
import helper
import matplotlib.pyplot as plt
import pdb

# Choose a mesh
if  False:
    # file_name = "lshape.xml"
    file_name = "dolfin_coarse.xml"
    #file_name = "dolfin_fine.xml"
    #file_name = "pinch.xml" 
    mesh_obj = Mesh( "meshes/" + file_name )
else:
    file_name = "square"
    mesh_obj = UnitSquareMesh( 20, 20 )

container = helper.Container( mesh_obj, 2, 1.23 )
kappa = helper.Kappa( container )
kappa.newton( iterations = 100 )

# Modify the right hand side vector to account for point source(s)
if file_name == "dolfin_coarse.xml" or file_name == "dolfin_fine.xml":
    pts = [ np.array( [ 0.45 , 0.65 ] ) ]
   
elif file_name == "pinch.xml":
    pts = [ np.array( [ 0.35 , 0.155 ] ) ]

elif file_name == "square":
    pts = [ np.array( [ 0.45 , 0.65 ] ),
            np.array( [ 0.995, 0.5  ] ),
            np.array( [ 0.05 , 0.005] )]
    
elif file_name == "lshape.xml":
    pts = [ np.array( [ 0.45 , 0.65 ] ),
            np.array( [ 0.995, 0.2  ] ),
            np.array( [ 0.05 , 0.005] )]
    
kappa.plot( pts )
