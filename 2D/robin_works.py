from dolfin import *
import numpy as np
import helper
import matplotlib.pyplot as plt
import pdb

if True:
    file_name = "dolfin_coarse.xml"
    #file_name = "dolfin_fine.xml"
    #file_name = "pinch.xml" 
    mesh_obj = Mesh( file_name )
else:
    file_name = ""
    mesh_obj = UnitSquareMesh( 50, 50 )

kappa = 1.
normal = FacetNormal( mesh_obj )

V = FunctionSpace ( mesh_obj, "CG", 1 )
u = TrialFunction ( V )
v = TestFunction ( V )
f = Constant( 0.0 )
  
beta = helper.Beta( kappa, mesh_obj, V )
 
# Sample variational forms
a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + inner( beta, normal )*u*v*ds
L = f*v*dx

A = assemble(a)
b = assemble(L)

# Modify the right hand side vector to account for point source(s)
if file_name == "dolfin_coarse.xml" or file_name == "dolfin_fine.xml":
    PointSource ( V, Point ( 0.45 , 0.65  ), 10. ).apply( b )
    PointSource ( V, Point ( 0.5  , 0.005 ), 10. ).apply( b )
    PointSource ( V, Point ( 0.5  , 0.995 ), 10. ).apply( b )
    PointSource ( V, Point ( 0.005, 0.5   ), 10. ).apply( b )
    PointSource ( V, Point ( 0.995, 0.5   ), 10. ).apply( b )

elif file_name == "pinch.xml":
    PointSource ( V, Point ( 0.05 , 0.005 ), 10. ).apply( b )
    PointSource ( V, Point ( 0.995, 0.1   ), 10. ).apply( b )
    PointSource ( V, Point ( 0.95 , 0.995 ), 10. ).apply( b )
    PointSource ( V, Point ( 0.005, 0.95  ), 10. ).apply( b )
    
else:
    PointSource ( V, Point ( 0.45 , 0.65  ), 10. ).apply( b )
    PointSource ( V, Point ( 0.995, 0.5   ), 10. ).apply( b )
    PointSource ( V, Point ( 0.05 , 0.005 ), 10. ).apply( b )

#  Solve the linear system for u.
sol = Function( V )
solve(A, sol.vector(), b, 'lu')

file = File( "vis.pvd")
file << sol
    
# Plot the solution.
#plot ( sol, interactive = True ) 
 
# pdb.set_trace()

fail = np.array( beta.fail )
if len(fail) > 3:
    fx = fail[:,0]
    fy = fail[:,1]
    plt.scatter( fx, fy, color = "red" )

win  = np.array( beta.win )
if len(win) > 3:    
    wx = win[:,0]
    wy = win[:,1]
    plt.scatter( wx, wy, color = "green" )
plt.show()
