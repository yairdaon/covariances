import numpy as np
from scipy import special as sp
from scipy.linalg import sqrtm as sqrtm
from dolfin import *
import pdb
import math

import container
import helper

class Mixed(Expression):

    def __init__( self, container, normal_run = True ):

        self.normal_run = normal_run
        self.container = container
        self.dic = {}
        
    def eval( self, value, y ):
        
       
        if self.dic.has_key( (y[0], y[1], y[2] ) ):
            t =  self.dic[ ( y[0], y[1], y[2] ) ]
            if self.normal_run:
                value[0] = t[0]
                value[1] = t[1]
                value[2] = t[2]
            else:
                value[0] = t
        else:
            normal = helper.cube_normal( y )
            if not ( self.normal_run or normal ):
                value[0] = 0.0
                return
                
            V = self.container.V
            kappa = self.container.kappa
            x  = self.container.x
            
            y_x = y-x
            ra  = y_x * y_x
            ra  = np.sum( ra, axis = 1 )
            ra  = np.sqrt( ra ) + 1e-13
            kappara = kappa * ra 
            

            # factor = K_0.5( r ) * e^{-r} / r
            factor_arr = sp.kv( 0.5, kappara ) * np.exp( -kappara ) * np.power( kappara , -.5 ) 
            factor = Function( V )           
            factor.vector().set_local( factor_arr[dof_to_vertex_map(V)] )
            
            # Get the denominator
            denom = 2.0 * assemble( factor * dx ) 

            # First, second and third components of enumerator
            enum_arr = ( 2.0 + np.power( kappara,-1) ) * np.power( kappara, -1 )
        
            enum_right_arr0 = enum_arr * y_x[:,0] 
            enum_right0     = Function( V )
            enum_right0.vector().set_local( enum_right_arr0[dof_to_vertex_map(V)] )
            enum0 = kappa * kappa * assemble( factor * enum_right0 * dx )
        
            enum_right_arr1 = enum_arr * y_x[:,1] 
            enum_right1     = Function( V )
            enum_right1.vector().set_local( enum_right_arr1[dof_to_vertex_map(V)] )
            enum1 = kappa * kappa * assemble( factor * enum_right1 * dx )
        
            enum_right_arr2 = enum_arr * y_x[:,2] 
            enum_right2     = Function( V )
            enum_right2.vector().set_local( enum_right_arr2[dof_to_vertex_map(V)] )
            enum2 = kappa * kappa * assemble( factor * enum_right2 * dx )
       

            if self.normal_run:
                value[0] = enum0/ denom
                value[1] = enum1/ denom
                value[2] = enum2/ denom
                self.dic[(y[0], y[1], y[2])] = (value[0], value[1], value[2]) 

            else:
                value[0] = ( normal[0]*enum0 + normal[1]*enum1 + normal[2]*enum2 ) /denom
                self.dic[(y[0], y[1], y[2])] = value[0]
 

    def value_shape( self ):
        return  (3,)
