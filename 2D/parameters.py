import numpy as np
from scipy import special as sp
from scipy.linalg import sqrtm as sqrtm
from dolfin import *
import pdb
import math

import helper
class Container():

    def __init__( self,
                  mesh_name,
                  mesh_obj,
                  kappa,
                  gamma = 1.0,
                  num_samples = 0,
                  sqrt_M = None,
                  M = None ):

        self.mesh_name = mesh_name
        self.dim = mesh_obj.geometry().dim()
        self.mesh_obj = mesh_obj
        self.normal = FacetNormal( mesh_obj )
        self.V  =       FunctionSpace( mesh_obj, "CG", 1 )
        self.V2 = VectorFunctionSpace( mesh_obj, "CG", 1 )
       
        # Not sure if need this...
        # self.R  = VectorFunctionSpace( mesh_obj, 'R' , 0 )
        # self.c  = TestFunction( self.R )

        self.u = TrialFunction( self.V )
        self.v = TestFunction( self.V )
        self.kappa  = kappa
        self.kappa2 = kappa**2
        self.gamma  = gamma
        self.n = self.V.dim()
       
        self._sqrt_M = sqrt_M
        self._M = M
        self._variances = {}
        self._gs = {}

        self.num_samples = num_samples
        
        self.set_constants()

    def generate( self, name ):
        file = open( "cpp/" + name + ".cpp" , 'r' )  
        code = file.read()
        
        if "enum" in name:
            xp = Expression( code, element = self.V2.ufl_element() )
        else:
            xp = Expression( code, element = self.V.ufl_element() )
            
        xp.kappa  = self.kappa / math.sqrt( self.gamma )
        xp.factor = self.factor
        return xp

    
    def set_constants( self ):
                
        gamma = self.gamma
        
        # We factor out gamma, so we scale kappa
        # accordingly. Later we compensate
        kappa2 = self.kappa2 / gamma
        
        # Here we compensate - we now have covariance
        # [ gamma * (-Delta + kappa^2 / gamma ) ]^2
        self.sig2 = gamma**2 / 4.0 / math.pi / kappa2 
        self.sig  = math.sqrt( self.sig2 )
        self.factor = self.sig2
        
        self.ran_var = ( 0.0, 4 * self.sig2 )
        self.ran_sol = ( 0.0, 2 * self.sig2 )

    @property
    def sqrt_M( self ):
        if self._sqrt_M == None:
            print "Preparing square root of mass matrix. This will take some time."
            self._sqrt_M =  sqrtm( self.M )
            print "Done!"
        return self._sqrt_M

    @property
    def M( self ):
        if self._M == None:
            self._M = assemble( self.u*self.v*dx )
        return self._M

    def gs( self, BC ):
        if BC in self._gs:
            return self._gs[BC]
        else:
            helper.set_vg( self, BC )
            return self._gs[BC]
            
    def variances( self, BC ):
        if BC in self._variances:
            return self._variances[BC]
        else:
            helper.set_vg( self, BC )
            return self._variances[BC]
        

class Robin(Expression):

    def __init__( self, container, enum, denom  ):
        
        self.container = container
              
        self.enum  =  self.container.generate( enum  )
        self.denom =  self.container.generate( denom )
              
        self.dic = {}
    
    def eval( self, value, x ):
        
        if self.dic.has_key( ( x[0],x[1] ) ):
            t = self.dic[ ( x[0], x[1] ) ]
            value[0] = t[0]
            value[1] = t[1]
        else:
            self.update( x )
            
            fe_denom = interpolate( self.denom, self.container.V ) 
            denom = assemble( fe_denom * dx )
                
            fe_enum = interpolate( self.enum, self.container.V2 )
            enum = assemble( dot(fe_enum, self.container.c) * dx )
                        
            value[0] = enum[0]/denom
            value[1] = enum[1]/denom

            self.dic[ ( x[0],x[1] )] = ( value[0], value[1] )
       
    def update( self, x ):

        self.x = x
        
        helper.update_x_xp( x, self.denom )
        helper.update_x_xp( x, self.enum )
                
   
    def value_shape(self):
        return (2,)
