#!/usr/bin/python
from dolfin import *
import numpy as np
import helper
import matplotlib.pyplot as plt
import pdb

# Choose a mesh
if  True:
    # file_name = "lshape.xml"
    file_name = "dolfin_coarse.xml"
    #file_name = "dolfin_fine.xml"
    #file_name = "pinch.xml" 
    mesh_obj = Mesh( "meshes/" + file_name )
else:
    file_name = "square"
    mesh_obj = UnitSquareMesh( 4, 4 )

kappa = 3.13
container = helper.Container( mesh_obj, 2, kappa )

robin  = helper.Robin( container )


# For notational convenience
V = container.V 

# All the pbjects used in the variational formulation below
normal = FacetNormal( mesh_obj )
u = container.u
v = container.v
f = Constant( 0.0 )

# Squared Robin Variational formulati
robin( 0.2, 0.3 )
pdb.set_trace()
L = f*v*dx + inner( robin, normal )[1]*v*ds
b = assemble(L)

a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + inner( robin, normal )[0]*u*v*ds

A = assemble(a)


# Modify the right hand side vector to account for point source(s)
if file_name == "dolfin_coarse.xml" or file_name == "dolfin_fine.xml":
    delta = PointSource ( V, Point ( 0.45 , 0.65  ), 10. )
    delta.apply( b )
    
elif file_name == "pinch.xml":
    delta = PointSource ( V, Point ( 0.35 , 0.155  ), 4. )
    delta.apply( b )
    

elif file_name == "square":
    PointSource ( V, Point ( 0.45 , 0.65  ), 10. ).apply( b )
    PointSource ( V, Point ( 0.995, 0.5   ), 10. ).apply( b )
    PointSource ( V, Point ( 0.05 , 0.005 ), 10. ).apply( b )
    
    
elif file_name == "lshape.xml":
    PointSource ( V, Point ( 0.45 , 0.65  ), 10. ).apply( b )
    PointSource ( V, Point ( 0.995, 0.2   ), 10. ).apply( b )
    PointSource ( V, Point ( 0.05 , 0.005 ), 10. ).apply( b )
   
    
tmp = Function( V )
# Solve the linear systems - Robin.
sol = Function( V )
solve(A, tmp.vector(), b, 'lu')
solve(A, sol.vector(), tmp.vector(), 'lu')

#Plot the solutions.
plot ( sol,               interactive = True, title = "Robin Solution (squared)" ) 


