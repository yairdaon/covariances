#!/usr/bin/python
from dolfin import *
import numpy as np
import helper
import matplotlib.pyplot as plt
import pdb

# Choose a mesh
if  True:
    # file_name = "lshape.xml"
    #file_name = "dolfin_coarse.xml"
    file_name = "dolfin_fine.xml"
    #file_name = "pinch.xml" 
    mesh_obj = Mesh( file_name )
else:
    file_name = "square"
    mesh_obj = UnitSquareMesh( 4, 4 )

# Choose kappa
kappa = 1.23

# Make the boundary object with all the required
# spaces etc.
beta = helper.Beta( kappa, mesh_obj , 2)

# For notational convenience
V = beta.V 

# All the pbjects used in the variational formulation below
normal = FacetNormal( mesh_obj )
u = TrialFunction ( beta.V )
v = TestFunction ( beta.V )
f = Constant( 0.0 )
  
# Robin Variational formulation
a_robin = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + inner( beta, normal )*u*v*ds
L_robin = f*v*dx
A_robin = assemble(a_robin)
b_robin = assemble(L_robin)

# Neumann Variational formulation
a_neumann = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx
L_neumann = f*v*dx
A_neumann = assemble(a_neumann)
b_neumann = assemble(L_neumann)

# Modify the right hand side vector to account for point source(s)
if file_name == "dolfin_coarse.xml" or file_name == "dolfin_fine.xml":
    delta = PointSource ( V, Point ( 0.45 , 0.65  ), 10. )
    delta.apply( b_robin )
    delta.apply( b_neumann )
   
elif file_name == "pinch.xml":
    delta = PointSource ( V, Point ( 0.35 , 0.155  ), 4. )
    delta.apply( b_robin )
    delta.apply( b_neumann )

elif file_name == "square":
    PointSource ( V, Point ( 0.45 , 0.65  ), 10. ).apply( b_robin )
    PointSource ( V, Point ( 0.995, 0.5   ), 10. ).apply( b_robin )
    PointSource ( V, Point ( 0.05 , 0.005 ), 10. ).apply( b_robin )
    PointSource ( V, Point ( 0.45 , 0.65  ), 10. ).apply( b_neumann )
    PointSource ( V, Point ( 0.995, 0.5   ), 10. ).apply( b_neumann )
    PointSource ( V, Point ( 0.05 , 0.005 ), 10. ).apply( b_neumann )
    
elif file_name == "lshape.xml":
    PointSource ( V, Point ( 0.45 , 0.65  ), 10. ).apply( b_robin )
    PointSource ( V, Point ( 0.995, 0.2   ), 10. ).apply( b_robin )
    PointSource ( V, Point ( 0.05 , 0.005 ), 10. ).apply( b_robin )
    PointSource ( V, Point ( 0.45 , 0.65  ), 10. ).apply( b_neumann )
    PointSource ( V, Point ( 0.995, 0.2   ), 10. ).apply( b_neumann )
    PointSource ( V, Point ( 0.05 , 0.005 ), 10. ).apply( b_neumann )
    

# Solve the linear systems - Robin.
sol_robin = Function( V )
solve(A_robin, sol_robin.vector(), b_robin, 'lu')

# Solve the linear systems - Neumann.
sol_neumann = Function( V )
solve(A_neumann, sol_neumann.vector(), b_neumann, 'lu')
 
# Save to memory
file = File( "vis/robin.pvd")
file << sol_robin   
file = File( "vis/neumann.pvd")
file << sol_neumann
file = File( "vis/diff.pvd")
file << project( sol_neumann - sol_robin, V ) 

#Plot the solutions.
plot ( sol_robin, interactive = True, title = "Robin Solution" ) 
plot ( sol_neumann, interactive = True, title = "Neumann Solution" ) 
plot ( sol_neumann - sol_robin, interactive = True, title = "Neumann solution minus Robin solution" ) 
