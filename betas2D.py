import numpy as np
import hashlib 
import math

from dolfin import *

import helper
import container

class Beta(Expression):
    '''
    Calculates a precursor to our optimal beta.
    We still need to take inner product with the
    unit normal (FEniCS does that for us) and 
    ensure the result is bigger than zero (FEniCS
    takes care of tha too).
    '''
    def __init__( self, container, version ):
        
        # The container variable holds all required data,
        # parameters etc. See the documentation in module
        # container.py
        self.container = container
                
        # Create the expressions for the integrals of the 
        # enumrator(s) and denominator.
        self.enum0 = IntegratedExpression( container, version + "_enum0" )
        self.enum1 = IntegratedExpression( container, version + "_enum1" )
        self.denom = IntegratedExpression( container, version + "_denom" )

        # Create a dictionary that will hold all calculated
        # values, so we don't have to redo calculations.
        self.dic = {}
        
    def eval( self, value, y ):
        
        # Make the numpy buffer hashable
        #y.flags.writeable = False
        
        # If we already did the calculation, return the value...
        if self.dic.has_key( y.tostring() ):
            value = self.dic[ y.tostring() ]
        
        # Otherwise, compute it from scratch:
        else:
        
            # FEniCS takes an inner product with the unit
            # normal to obtain our beta.
            denom_variable = self.denom( y )
            value[0]  = self.enum0( y ) / denom_variable
            value[1]  = self.enum1( y ) / denom_variable

            # Keep track of what we calculated, so we don't have to do it again.
            self.dic[y.tostring()] = value

    def value_shape(self):
        return (2,)     
       

class Mix3D(Expression):
    '''
    Calculates a precursor to our optimal beta.
    We still need to take inner product with the
    unit normal (FEniCS does that for us) and 
    ensure the result is bigger than zero (FEniCS
    takes care of tha too).
    '''
    def __init__( self, container, version ):
        
        # The container variable holds all required data,
        # parameters etc. See the documentation in module
        # container.py
        self.container = container
                
        # Create the expressions for the integrals of the 
        # enumrator(s) and denominator.
        self.enum0 = IntegratedExpression( container, version + "_enum0" )
        self.enum1 = IntegratedExpression( container, version + "_enum1" )
        self.enum2 = IntegratedExpression( container, version + "_enum2" )
        self.denom = IntegratedExpression( container, version + "_denom" )

        # Create a dictionary that will hold all calculated
        # values, so we don't have to redo calculations.
        self.dic = {}
        
    def eval( self, value, y ):

        # Make the numpy buffer hashable
        #y.flags.writeable = False
        
        # If we already did the calculation, return the value...
        if self.dic.has_key( y.tostring() ):
            value = self.dic[ y.tostring() ]
        
        # Otherwise, compute it from scratch:
        else:
        
            # FEniCS takes an inner product with the unit
            # normal to obtain our beta.
            denom_variable = self.denom( y )
            value[0]  = self.enum0( y ) / denom_variable
            value[1]  = self.enum1( y ) / denom_variable
            value[2]  = self.enum2( y ) / denom_variable

            # Keep track of what we calculated, so we don't have to do it again.
            self.dic[ y.tostring() ] = value

    def value_shape(self):
        return (3,)


class IntegratedExpression(Expression):
    '''
    Compile and integrate a given expression. Expression
    is given as a string that refers to a c++ file
    which is then compiled.
    '''
    
    def __init__( self, container, expression_string ):
        
        # The container variable holds all required data,
        # parameters etc. See the documentation in module
        # container.py
        self.container = container
        
        # Generate the expression from the c++ file that
        # the input variable expresion_string refers to.
        xp = self.generate( expression_string )
            
        xp.kappa  = container.kappa / math.sqrt( container.gamma )
        xp.factor = container.factor
        self.expression = xp

        # Create a dictionary that will hold all calculated
        # values, so we don't have to redo calculations.
        self.dic = {}
    
    def eval( self, value, y ):
        
        # If we already did the calculation, return the value...
        if self.dic.has_key( y.tostring() ):
            value = self.dic[ y.tostring() ]

        # Otherwise, compute it from scratch:
        else:

            # Let the expression know with respect to which (boundary)
            # point we want its value.
            for i in range( self.container.dim ):
                self.expression.y[i] = y[i]
            
            # The return value is the an approximation to the integral of the expression
            value[0] = assemble( 
                self.expression * dx(self.container.mesh_obj )#, metadata={'quadrature_degree': 7} )
                )
            # Keep track of what we calculated, so we don't have to do it again.
            self.dic[ y.tostring() ] = value


    def generate( self, expression_string ):
        '''
        Generate the expression from the c++ file that
        the input variable refers to.
        '''
        loc_file = open( "cpp/" + expression_string + ".cpp" , 'r' )  
        code = loc_file.read()
        return Expression( code, degree = 3 ) 
