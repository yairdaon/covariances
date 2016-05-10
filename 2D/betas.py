import numpy as np
from scipy import special as sp
from scipy.linalg import sqrtm as sqrtm
from dolfin import *
import pdb
import math

import helper
import container

class Beta(Expression):

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
