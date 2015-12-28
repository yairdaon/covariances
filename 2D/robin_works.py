from dolfin import *
import numpy
import helper

kappa = 1.23

m = n = 20
mesh = UnitSquareMesh( m-1, n-1 )

V = FunctionSpace ( mesh, "CG", 1 )
u = TrialFunction ( V )
v = TestFunction ( V )
f = Constant( .0 )
  
beta = helper.Beta( kappa, mesh, V )
      
# Sample variational forms
a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + beta*u*v*ds
L = f*v*dx

A = assemble(a)
b = assemble(L)

# Modify the right hand side vector to account for point source(s)
delta = PointSource ( V, Point ( 0.01, 0.5 ), 10.0 )
delta.apply ( b )
gamma = PointSource ( V, Point( 0.95, 0.05 ), 10.0 )
gamma.apply( b )
eta = PointSource ( V, Point( 0.5, 0.5 ), 10.0 )
eta.apply( b )

#  Solve the linear system for u.
u = Function( V )
solve(A, u.vector(), b, 'lu')

file = File( "vis.pvd")
file << u
    
# Plot the solution.
plot ( u, interactive = True ) 
