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

# Make the boundary object with all the required
# spaces etc.
beta  = helper.Beta( container, squared = False )
beta2 = helper.Beta( container, squared = True )

# For notational convenience
V = container.V 

# All the pbjects used in the variational formulation below
normal = FacetNormal( mesh_obj )
u = container.u
v = container.v
f = Constant( 0.0 )
  
# Robin Variational formulation
a_robin = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + inner( beta, normal )*u*v*ds
L_robin = f*v*dx
A_robin = assemble(a_robin)
b_robin = assemble(L_robin)


# Squared Robin Variational formulation
a_robin2 = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + inner( beta2, normal )*u*v*ds
L_robin2 = f*v*dx
A_robin2 = assemble(a_robin2)
b_robin2 = assemble(L_robin2)

# Neumann Variational formulation
a_neumann = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx
L_neumann = f*v*dx
A_neumann = assemble(a_neumann)
b_neumann = assemble(L_neumann)

# Modify the right hand side vector to account for point source(s)
if file_name == "dolfin_coarse.xml" or file_name == "dolfin_fine.xml":
    delta = PointSource ( V, Point ( 0.45 , 0.65  ), 10. )
    delta.apply( b_robin )
    delta.apply( b_robin2 )
    delta.apply( b_neumann )
   
elif file_name == "pinch.xml":
    delta = PointSource ( V, Point ( 0.35 , 0.155  ), 4. )
    delta.apply( b_robin )
    delta.apply( b_robin2 )
    delta.apply( b_neumann )

elif file_name == "square":
    PointSource ( V, Point ( 0.45 , 0.65  ), 10. ).apply( b_robin )
    PointSource ( V, Point ( 0.995, 0.5   ), 10. ).apply( b_robin )
    PointSource ( V, Point ( 0.05 , 0.005 ), 10. ).apply( b_robin )
    PointSource ( V, Point ( 0.45 , 0.65  ), 10. ).apply( b_robin2 )
    PointSource ( V, Point ( 0.995, 0.5   ), 10. ).apply( b_robin2 )
    PointSource ( V, Point ( 0.05 , 0.005 ), 10. ).apply( b_robin2 )
    PointSource ( V, Point ( 0.45 , 0.65  ), 10. ).apply( b_neumann )
    PointSource ( V, Point ( 0.995, 0.5   ), 10. ).apply( b_neumann )
    PointSource ( V, Point ( 0.05 , 0.005 ), 10. ).apply( b_neumann )
    
elif file_name == "lshape.xml":
    PointSource ( V, Point ( 0.45 , 0.65  ), 10. ).apply( b_robin )
    PointSource ( V, Point ( 0.995, 0.2   ), 10. ).apply( b_robin )
    PointSource ( V, Point ( 0.05 , 0.005 ), 10. ).apply( b_robin )
    PointSource ( V, Point ( 0.45 , 0.65  ), 10. ).apply( b_robin2 )
    PointSource ( V, Point ( 0.995, 0.2   ), 10. ).apply( b_robin2 )
    PointSource ( V, Point ( 0.05 , 0.005 ), 10. ).apply( b_robin2 )
    PointSource ( V, Point ( 0.45 , 0.65  ), 10. ).apply( b_neumann )
    PointSource ( V, Point ( 0.995, 0.2   ), 10. ).apply( b_neumann )
    PointSource ( V, Point ( 0.05 , 0.005 ), 10. ).apply( b_neumann )
    
tmp = Function( V )
# Solve the linear systems - Robin.
sol_robin = Function( V )
solve(A_robin, tmp.vector(), b_robin, 'lu')
solve(A_robin, sol_robin.vector(), tmp.vector(), 'lu')


# Solve the linear systems - Robin squared.
sol_robin2 = Function( V )
solve(A_robin2, tmp.vector(), b_robin2, 'lu')
solve(A_robin2, sol_robin2.vector(), tmp.vector(), 'lu')

# Solve the linear systems - Neumann.
sol_neumann = Function( V )
solve(A_neumann, tmp.vector(), b_neumann, 'lu')
solve(A_neumann, sol_neumann.vector(), tmp.vector(), 'lu')
 
# Save to memory
file = File( "vis/robin.pvd")
file << sol_robin   
file = File( "vis/robin2.pvd")
file << sol_robin2   
file = File( "vis/neumann.pvd")
file << sol_neumann
file = File( "vis/diff.pvd")
file << project( sol_neumann - sol_robin, V ) 

#Plot the solutions.
plot ( sol_robin,                interactive = True, title = "Robin Solution" ) 
plot ( sol_robin2,               interactive = True, title = "Robin Solution (squared)" ) 
plot ( sol_neumann,              interactive = True, title = "Neumann Solution" ) 
plot ( sol_neumann - sol_robin,  interactive = True, title = "Neumann solution minus Robin solution" ) 
plot ( sol_neumann - sol_robin2, interactive = True, title = "Neumann solution minus Robin solution2" )
plot ( sol_robin   - sol_robin2, interactive = True, title = "Robin solution minus Robin solution2" ) 

