import numpy as np
from scipy import special as sp
from dolfin import *
import pdb

class Beta(Expression):

    def __init__( self, kappa, mesh, deg ):
        
        self.mesh = mesh
        self.V = FunctionSpace( mesh, "CG", deg )
        self.V2 = VectorFunctionSpace( mesh, "CG", deg, dim=2 )
        self.kappa = kappa
        
        G_file = open( "G.cpp" , 'r' )  
        G_code = G_file.read()
        self.G = Expression( G_code )
        self.G.kappa = kappa

        gradG_file = open( "gradG.cpp" , 'r' )  
        gradG_code = gradG_file.read()
        self.gradG = Expression( gradG_code )
        self.gradG.kappa = kappa
    
    def value_shape(self):
        return (2,)
    
    def eval(self, value, x ):
        '''
        x is a point on the boundary for which we calculate beta according to:
        
                  \int \nabla_x G(x,y) * G(x,y) dy
                  \Omega
        beta(y) = --------------------------------
    
                  \int G^2(x,y) dy
                  \Omega

        where the normal is outward at y and differentiation is wrt y
        '''
        
        
        self.update_x( x )
        fe_G      = project( self.G,      self.V2  ) 
        fe_gradG  = project( self.gradG , self.V2 )
    
        denominator  = assemble( fe_G[0] * fe_G[0]     * dx )
        enumerator0  = assemble( fe_G[0] * fe_gradG[0] * dx )
        enumerator1  = assemble( fe_G[1] * fe_gradG[1] * dx )

        value[0] = -enumerator0 / denominator
        value[1] = -enumerator1 / denominator
            
    def update_x( self, x ):

        self.x = x

        self.G.x[0] = x[0]
        self.G.x[1] = x[1]

        self.gradG.x[0] = x[0]
        self.gradG.x[1] = x[1]
