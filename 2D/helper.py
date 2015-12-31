import numpy as np
from scipy import special as sp
from dolfin import *
import pdb

class Beta(Expression):

    def __init__( self, kappa, mesh, V ):
        
        self.V = V
        self.mesh = mesh
        self.V2 = VectorFunctionSpace( mesh, "CG", 1, dim=2 )
        self.kappa = kappa
        
        G_file = open( "G.cpp" , 'r' )  
        G_code = G_file.read()
        self.G = Expression( G_code )
        self.G.kappa = kappa

        gradG_file = open( "gradG.cpp" , 'r' )  
        gradG_code = gradG_file.read()
        self.gradG = Expression( gradG_code )
        self.gradG.kappa = kappa
        
        self.fail = []
        self.den = []
        self.win = []
    
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

        if denominator == 0.0:
            self.fail.append( np.array([ x[0], x[1] ]) )
            self.den.append( denominator )
            value[0] = 0.0
            value[1] = 0.0
        else:
            self.win.append (np.array( [ x[0] , x[1] ] ) )
            value[0] = -enumerator0 / denominator
            value[1] = -enumerator1 / denominator
            
    def update_x( self, x ):

        self.G.x[0] = x[0]
        self.G.x[1] = x[1]

        self.gradG.x[0] = x[0]
        self.gradG.x[1] = x[1]
