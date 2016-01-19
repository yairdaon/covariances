#!/usr/bin/python
from dolfin import *
import numpy as np

import matplotlib.pyplot as plt
import pdb

n = 201                          
mesh_obj = UnitIntervalMesh( n-1 )

    
# For notational convenience
V = FunctionSpace( mesh_obj, "CG", 1 )
                                                   
dof_coo = V.dofmap().tabulate_all_coordinates(mesh_obj)                      
dof_coo.resize((n,))              
#print dof_coo

# All the function definitions we need
u     = TrialFunction ( V )
v     = TestFunction ( V )
sol_full   = Function( V )
sol_partial = Function( V )

# The killing rate
kappa_full = Function( V )
kappa_full.vector()[:] = 1.25

kappa_partial = Function( V )
kappa_partial.vector()[:] = 1.25


# Holds pointwise variance values
var_full   = Function( V )
var_partial = Function( V )


ff = Constant( 0.0 )
LL = ff*v*dx
bb = assemble( LL )
PointSource( V, Point(0.1), 1. ).apply( bb )

G = np.empty( (n,n) )


counter = 1
while True:
    counter = counter + 1
    print counter    
    
    a_full = inner(grad(u), grad(v))*dx + kappa_full*u*v*dx 
    A_full = assemble(a_full)
    
    a_partial = inner(grad(u), grad(v))*dx + kappa_partial*u*v*dx 
    A_partial = assemble(a_partial)
    
    for i in range(0,n):
        
        pt = Point( dof_coo[i] )
        f = Constant( 0.0 )
        L = f*v*dx
        b = assemble( L )
        PointSource( V, pt, 1. ).apply( b )
    
        solve(A_full, sol_full.vector(), b, 'lu')
        G[i,:] = sol_full.vector()
        var_full.vector()[i] = sol_full.vector()[i][0]
        
        solve(A_partial, sol_partial.vector(), b, 'lu')
        var_partial.vector()[i] = sol_partial.vector()[i][0]
        
    kappa_partial.vector()[:] = kappa_partial.vector()[:] + ( var_partial.vector() - 1 ) * np.power( var_partial.vector(),-2 )
    kappa_full.vector()[:] = kappa_full.vector()[:] + np.linalg.solve( G * np.transpose( G ),  var_full.vector() - 1 ) 
    
    if counter % 1000 == 0:
        
        solve(A_full, sol_full.vector(), bb, 'lu')
        solve(A_partial, sol_partial.vector(), bb, 'lu')

        file = File( "vis/kappa_full.pvd")
        file << kappa_full 
        file = File( "vis/var_full.pvd")
        file << var_full
        file = File( "vis/sol_full.pvd")
        file << sol_full
        
        file = File( "vis/kappa_partial.pvd")
        file << kappa_partial 
        file = File( "vis/var_partial.pvd")
        file << var_partial
        file = File( "vis/sol_partial.pvd")
        file << sol_partial
       
        
        mx_full    = np.linalg.norm( var_full.vector() - 1.0, ord = np.inf )
        mx_partial = np.linalg.norm( var_partial.vector() - 1.0, ord = np.inf )
        print "Error for full    factorization = " + str(mx_full)
        print "Error for partial factorization = " + str(mx_partial)
        print
        if mx_full < 1E-4 or mx_partial < 1E-4:
            break
            
            
        if counter == 15000:
            break
    
    


#Plot the solutions.
plot( kappa_full, interactive = True, title = "kappa, full Df" ) 
plot( var_full,   interactive = True, title = "Variance, full Df" )
plot( sol_full,   interactive = True, title = "Solution, full Df" ) 

#Plot the solutions.
plot( kappa_partial, interactive = True, title = "kappa, partial Df" ) 
plot( var_partial,   interactive = True, title = "Variance, partial Df" )
plot( sol_partial,   interactive = True, title = "Solution, partial Df" ) 
