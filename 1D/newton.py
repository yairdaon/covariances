#!/usr/bin/python
from dolfin import *
import numpy as np

import matplotlib.pyplot as plt
import pdb

class Kappa():
    
    def __init__( self, mesh_obj, n ):
    
        self.n = n
        self.mesh_obj = mesh_obj
        
        # For notational convenience
        self.V = FunctionSpace( self.mesh_obj, "CG", 1 )
                                                   
        dof_coo = self.V.dofmap().tabulate_all_coordinates(self.mesh_obj)                      
        dof_coo.resize((n,))
        self.dof_coo = dof_coo

        self.var = Function( self.V )
        self.sol = Function( self.V )
        self.k   = Function( self.V )  
        self.u   = TrialFunction ( self.V )
        self.v   = TestFunction ( self.V )
        self.L   = Constant( 0.0 )*self.v*dx
        self.a   = inner(grad(self.u), grad(self.v))*dx + self.k*self.u*self.v*dx  
        self.src = assemble( self.L ) 
        PointSource( self.V, Point(0.1), 1. ).apply( self.src )
        self.k.vector()[:] = 1.25

    def stop( self, count ):
        
        if count % 500 == 0:
            mx    = np.linalg.norm( self.var.vector() - 1.0, ord = np.inf )
            print "Iteration " + str(count) + ". Variance Error = " + str(mx)
            print
            if mx < 1E-4:
                return True
            
        return False
        
    def plot(self):
         
        self.A = assemble( self.a )
        solve(self.A, self.sol.vector(), self.src, 'lu')
        
        plot( self.sol, interactive = True, title = "Solution" )         
        plot( self.var, interactive = True, title = "Variance" )
        plot( self.k,   interactive = True, title = "kappa" ) 

    def newton(self, iterations = 100, full = False):

        tmp = Function( self.V )
        
        if full:
            G = np.empty( (n,n) )
        
        for count in range(1,iterations):

            self.A = assemble(self.a)
           
            if self.stop(count):
                break
       
            for i in range( 0, self.n ):
        
                pt = Point( self.dof_coo[i] )
                b  = assemble( self.L )
                PointSource( self.V, pt, 1. ).apply( b )
    
                solve(self.A, tmp.vector(), b, 'lu')
                self.var.vector()[i] = tmp.vector()[i][0]
                if full:
                    G[i,:] = tmp.vector()
            
            if full:    
                self.k.vector()[:] = self.k.vector()[:] + np.linalg.solve( G * np.transpose( G ), self.var.vector() - 1 ) 
            else:
                self.k.vector()[:] = self.k.vector()[:] + ( self.var.vector() - 1 ) * np.power( self.var.vector(), -2 )                     
    
n = 51
mesh_obj = UnitIntervalMesh( n-1 )
kappa = Kappa( mesh_obj, n )
kappa.newton(iterations = 5000)
kappa.newton(full = True)
kappa.plot()
pdb.set_trace()
# file = File( "vis/kappa.pvd")
# file << kappa 
# file = File( "vis/var.pvd")
# file << var
# file = File( "vis/sol.pvd")
# file << sol

