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
    
    def eval( self, value, y ):
        
        if self.dic.has_key( ( y[0],y[1] ) ):
            t = self.dic[ ( y[0], y[1] ) ]
            value[0] = t[0]
            value[1] = t[1]
        else:
            self.update( y )
            
            fe_denom = interpolate( self.denom, self.container.V ) 
            denom = assemble( fe_denom * dx )
                
            fe_enum  = interpolate( self.enum, self.container.V2 )
            enum = assemble( dot(fe_enum, self.container.c) * dx )
                        
            value[0] = enum[0]/denom
            value[1] = enum[1]/denom

            self.dic[ ( y[0],y[1] )] = ( value[0], value[1] )
       
    def update( self, y ):

        self.y = y
        
        self.update_y_xp( y, self.denom )
        self.update_y_xp( y, self.enum )
                
   
    def update_y_xp( self, y, xp ):
        '''
        given a 2D expression xp, update its
        x variable.
        in terms of the paper, x is a boundary point
        which is denoted there by y
        '''
        xp.y[0] = y[0]
        xp.y[1] = y[1]

    def value_shape(self):
        return (2,)
