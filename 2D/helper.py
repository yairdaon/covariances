import numpy as np
from scipy import special as sp
from dolfin import *
import pdb

class Beta(Expression):

    def __init__( self, kappa, mesh, V ):
        
        self.V = V
        self.mesh = mesh
        self.kappa = kappa

        G_file = open( "G.cpp" , 'r' )  
        G_code = G_file.read()
        
        dGdn_file = open( "dGdn.cpp" , 'r' )  
        dGdn_code = dGdn_file.read()

        self.G = Expression( G_code )
        self.G.kappa = kappa
        
        self.dGdn = Expression( dGdn_code )
        self.dGdn.kappa = kappa

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
            fe_G    = project( self.G,    self.V )
            fe_dGdn = project( self.dGdn, self.V )
    
            denominator = assemble( fe_G * fe_G    * dx )
            enumerator  = assemble( fe_G * fe_dGdn * dx )
            value[0] = enumerator / denominator
        else:
            value[0] = 0.0

    def update_x( self, x ):

        self.x = x
        self.update_xn( x )
        self.G.x[0] = x[0]
        self.G.x[1] = x[1]

        self.dGdn.x[0] = x[0]
        self.dGdn.x[1] = x[1]

        self.dGdn.xn[0] = self.xn[0]
        self.dGdn.xn[1] = self.xn[1]
  
    def update_xn( self, x ):

        xn = np.zeros( (2,) )
    
        if abs( x[0] ) < DOLFIN_EPS:
            xn[0] = -1.0
        elif abs( x[0] - 1 ) < DOLFIN_EPS:
            xn[0] = 1.0
        
           
        if abs( x[1] ) < DOLFIN_EPS:
            xn[1] = -1.0
        elif abs( x[1] - 1 ) < DOLFIN_EPS:
            xn[1] = 1.0
    
        self.xn = xn / np.linalg.norm( xn )
