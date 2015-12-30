import numpy as np
from scipy import special as sp
from dolfin import *
import pdb

class Beta(Expression):

    def __init__( self, kappa, mesh, V ):
        
        self.V = V
        self.V2 = VectorFunctionSpace( mesh, "CG", 1, dim = 2 )
        self.mesh = mesh
        self.kappa = kappa
        
        G_file = open( "G.cpp" , 'r' )  
        G_code = G_file.read()
        self.G = Expression( G_code )
        self.G.kappa = kappa

        gradG0_file = open( "gradG0.cpp" , 'r' )  
        gradG0_code = gradG0_file.read()
        self.gradG0 = Expression( gradG0_code )
        self.gradG0.kappa = kappa

        gradG1_file = open( "gradG1.cpp" , 'r' )  
        gradG1_code = gradG1_file.read()
        self.gradG1 = Expression( gradG1_code )
        self.gradG1.kappa = kappa
    
    def value_shape(self):
        return (2,)
    
    def eval(self, value, x ):
        '''
        y is a point on the boundary for which we calculate beta according to:
        
                  \int dG / dn G(x,y) dx
                  \Omega
        beta(y) = --------------------------------
    
                  \int G^2(x,y) dx
                  \Omega

        where the normal is outward at y and differentiation is wrt y
        '''
        
        
        if (abs(x[0]) < DOLFIN_EPS or abs(x[0]-1) < DOLFIN_EPS) or (abs(x[1]) < DOLFIN_EPS or abs(x[1]-1) < DOLFIN_EPS):
          
            self.update_x( x )
            fe_G      = project( self.G,      self.V )
            fe_gradG0 = project( self.gradG0, self.V )
            fe_gradG1 = project( self.gradG1, self.V )
    
            denominator  = assemble( fe_G * fe_G    * dx )
            enumerator0  = assemble( fe_G * fe_gradG0 * dx )
            enumerator1  = assemble( fe_G * fe_gradG1 * dx )

            value[0] = enumerator0 / denominator
            value[1] = enumerator1 / denominator
        else:
            
            value[0] = 0.0
            value[1] = 0.0

    def update_x( self, x ):

        self.x = x
       
        self.G.x[0] = x[0]
        self.G.x[1] = x[1]

        self.gradG0.x[0] = x[0]
        self.gradG0.x[1] = x[1]

        self.gradG1.x[0] = x[0]
        self.gradG1.x[1] = x[1]
