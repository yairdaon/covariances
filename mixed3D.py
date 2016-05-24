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
            
            # Get the denominator
            denom_left_arr = 2.0 * sp.kv( 0.5, kappara )
            denom_left     = Function( V )
            denom_left.vector().set_local( denom_left_arr[dof_to_vertex_map(V)] )
           
            denom_right_arr = np.exp( -kappara ) / np.sqrt( kappara )
            denom_right     = Function( V )
            denom_right.vector().set_local( denom_right_arr[dof_to_vertex_map(V)] )

            denom = assemble( denom_left * denom_right * dx ) 

            # First, second and third components of enumerator
            enum_left_tmp  = kappa * kappa * sp.kv( 0.5, kappara ) * np.exp( -kappara )
            enum_right_tmp = ( 2.0 * kappara + 1.0 ) * np.power( kappara , -2.5 ) 
           
            enum_left_arr0 = enum_left_tmp
            enum_left0     = Function( V )
            enum_left0.vector().set_local( enum_left_arr0[dof_to_vertex_map(V)] )
            enum_right_arr0 = enum_right_tmp * y_x[:,0] 
            enum_right0     = Function( V )
            enum_right0.vector().set_local( enum_right_arr0[dof_to_vertex_map(V)] )
            enum0 = assemble( enum_left0 * enum_right0 * dx )
            
            enum_left_arr1 = enum_left_tmp
            enum_left1     = Function( V )
            enum_left1.vector().set_local( enum_left_arr1[dof_to_vertex_map(V)] )
            enum_right_arr1 = enum_right_tmp * y_x[:,1] 
            enum_right1     = Function( V )
            enum_right1.vector().set_local( enum_right_arr1[dof_to_vertex_map(V)] )
            enum1 = assemble( enum_left1 * enum_right1 * dx )
            
            enum_left_arr2 = enum_left_tmp
            enum_left2     = Function( V )
            enum_left2.vector().set_local( enum_left_arr2[dof_to_vertex_map(V)] )
            enum_right_arr2 = enum_right_tmp * y_x[:,2] 
            enum_right2     = Function( V )
            enum_right2.vector().set_local( enum_right_arr2[dof_to_vertex_map(V)] )
            enum2 = assemble( enum_left2 * enum_right2 * dx )

            if self.normal_run:
                value[0] = enum0/denom
                value[1] = enum1/denom
                value[2] = enum2/denom
                self.dic[(y[0], y[1], y[2])] = (value[0], value[1], value[2]) 

            else:
                value[0] = ( normal[0]*enum0 + normal[1]*enum1 + normal[2]*enum2 ) / denom
                self.dic[(y[0], y[1], y[2])] = value[0]
 

    def value_shape( self ):
        return  (3,)
