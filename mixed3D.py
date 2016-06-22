import numpy as np
from scipy import special as sp
from scipy.linalg import sqrtm as sqrtm
from dolfin import *
import pdb
import math

import container
import helper

class Mixed(Expression):
    '''
    a class that evaluates the beta we use in the 2nd
    variant of the Robin boundary condition. This class 
    does the calculation for 3D problems.
    '''
    def __init__( self, container, reg = 1e-12 ):
        
        # Holds pretty much every single possible
        # parameter, mesh, FunctionSpace etc.
        self.container = container
        
        # The regularization we use to ensure we don't evaluate
        # the integrand close to zero
        self.reg = reg

        # Store results of previous calculations,
        # so we don't need to perform them again.
        self.dic = {}
        
    def eval( self, value, y ):
        '''
        evaluate the expression. As mensioned before,
        this would evaluate to a 3-tuple of shape (3,).
        The way this is used can be seen in the module
        ordinary.py.
        '''
        
        # If we have the key, we don't need to perform
        # the lengthy calculation again.
        if self.dic.has_key( (y[0], y[1], y[2] ) ):
            value[0], value[1], value[2] = self.dic[ ( y[0], y[1], y[2] ) ]
            
        else:
                
            V = self.container.V
            kappa = self.container.kappa
            x  = self.container.x
            
            y_x = y-x
            ra  = y_x * y_x
            ra  = np.sum( ra, axis = 1 )
            ra  = np.sqrt( ra ) + self.reg
            kappara = kappa * ra 

            # factor = K_0.5( r ) * e^{-r} / r
            factor_arr = sp.kv( 0.5, kappara ) * np.exp( -kappara ) * np.power( kappara , -0.5 ) 
            factor = Function( V )           
            factor.vector().set_local( factor_arr[dof_to_vertex_map(V)] )
            
            # Get the denominator
            denom = 2.0 * assemble( factor * dx ) 

            enum_arr = ( 2.0 + np.power( kappara,-1.) ) * np.power( kappara, -1. )
                        
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
                   
            value[0] = enum0 / denom
            value[1] = enum1 / denom
            value[2] = enum2 / denom
            self.dic[(y[0], y[1], y[2])] = (value[0], value[1], value[2]) 

    def value_shape( self ):
        return  (3,)
