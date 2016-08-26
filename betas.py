import numpy as np
import hashlib 
import math

from dolfin import *
import instant

import helper
import container
import radial

class Beta2DAdaptive(Expression):
    '''
    Compile and integrate a given expression. Expression
    is given as a string that refers to a c++ file
    which is then compiled.
    '''
    
    def __init__( self, container, tol = 1e-9 ):
        
        # The container variable holds all required data,
        # parameters etc. See the documentation in module
        # container.py
        self.container = container
        
        # Generate the expression from the c++ file that
        # the input variable expresion_string refers to.
        self.xpr = generateInstant( "beta2D" )
            
        # Create a dictionary that will hold all calculated
        # values, so we don't have to redo calculations.
        self.dic = {}

        self.coordinates = container.mesh_obj.coordinates()
        self.elements = dict((cell.index(), cell.entities(0)) for cell in cells(container.mesh_obj))
    
        self.tol = tol
        
    def eval( self, value, y ):
        
        # If we already did the calculation, return the value...
        if self.dic.has_key( y.tostring() ):
            value = self.dic[ y.tostring() ]

        # Otherwise, compute it from scratch:
        else:
            
            # the dimension
            d = self.container.dim

            result = np.zeros( (d+1,) )
            vertices = np.empty(  (d+1)*d  )
     
            for element in range(self.container.mesh_obj.num_cells()):
                    i = 0
                    for vertex in self.elements[element]:
                        vertices[i:i+d] = self.coordinates[vertex]
                        i = i + d
                    self.xpr.integrateVector(  y, self.container.kappa, vertices, self.tol, result )  
                
            # The last entry is the denominator, by which we divide.
            # The first entries are the enumerators, which is a vector
            # that will later be multiplied by a unit normal
            for i in range( d ):
                value[i] = result[i] / result[d]


            # Keep track of what we calculated, so we don't have to do it again.
            self.dic[ y.tostring() ] = value



    # Should change this to apply to three dimensions!!!
    def value_shape(self):
        return (2,)
    

def generateInstant( expression_string ):
    '''
    Generate the expression from the c++ file that
    the input variable refers to.
    '''
        
    xpr_file = open( "integrate_vector.cpp" , 'r' )  
    xpr_code = xpr_file.read()
    xpr_file.close()
    
    # Replace the function we use into the text of the c++ file.. 
    xpr_code = xpr_code.replace( "FUNCTION_NAME", expression_string )
        
    # Compile and return an object with "integrate vector" method
    xpr = instant.build_module(
        code = xpr_code,
        sources = [ "helper.cpp", "hcubature.c" ],
        system_headers=["numpy/arrayobject.h"],
        include_dirs=[np.get_include()],
        init_code="import_array();",
        local_headers = [ "header.h", "cubature.h" ],
        arrays=[[ "n", "y"        ],
                [ "m", "vertices" ],
                [ "fdim", "result"   ]]
        )
    print "Beta Instant Compiled successfully!"
    return xpr
             


class BetaCubeAdaptive(Expression):
    '''
    Compile and integrate a given expression. Expression
    is given as a string that refers to a c++ file
    which is then compiled.
    '''
    
    def __init__( self, container, tol = 1e-9 ):
        
        # The container variable holds all required data,
        # parameters etc. See the documentation in module
        # container.py
        self.container = container
        
        # Generate the expression from the c++ file that
        # the input variable expresion_string refers to.
        
        # Read the code from file
        xpr_file = open( "integrate_cube.cpp" , 'r' )  
        xpr_code = xpr_file.read()
        xpr_file.close()
                
        # Compile and return an object with "integrate vector" method
        self.xpr = instant.build_module(
            code = xpr_code,
            sources = [ "helper.cpp", "hcubature.c" ],
            system_headers=["numpy/arrayobject.h"],
            include_dirs=[np.get_include()],
            init_code="import_array();",
            local_headers = [ "header.h", "cubature.h" ],
            arrays=[[ "n"   , "y"        ],
                    [ "fdim", "result"   ]]
            )
        print "Beta Cube Instant Compiled successfully!"
                
        # Create a dictionary that will hold all calculated
        # values, so we don't have to redo calculations.
        self.dic = {}
        
        self.tol = tol
        
    def eval( self, value, y ):
        
        # If we already did the calculation, return the value...
        if self.dic.has_key( y.tostring() ):
            value = self.dic[ y.tostring() ]

        # Otherwise, compute it from scratch:
        else:
            
            result = np.empty( 4 )

            self.xpr.integrateVectorOnCube( y, 
                                            self.container.kappa, 
                                            self.tol, 
                                            result )  
            
            # The last entry is the denominator, by which we divide.
            # The first entries are the enumerators, which is a vector
            # that will later be multiplied by a unit normal
            for i in range( 3 ):
                value[i] = result[i] / result[3]


            # Keep track of what we calculated, so we don't have to do it again.
            self.dic[ y.tostring() ] = value

    def value_shape(self):
        return (3,)
    
class Beta2D(Expression):
    '''
    Calculates a precursor to our optimal beta.
    We still need to take inner product with the
    unit normal (FEniCS does that for us) and 
    ensure the result is bigger than zero (FEniCS
    takes care of tha too).
    '''
    def __init__( self, container ):
        
        # The container variable holds all required data,
        # parameters etc. See the documentation in module
        # container.py
        self.container = container
                
        # Create the expressions for the integrals of the 
        # enumrator(s) and denominator.
        self.enum0 = IntegratedExpression( container, "enum0_2d" )
        self.enum1 = IntegratedExpression( container, "enum1_2d" )
        self.denom = IntegratedExpression( container, "denom_2d" )

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
       

class Beta3D(Expression):
    '''
    Calculates a precursor to our optimal beta.
    We still need to take inner product with the
    unit normal (FEniCS does that for us) and 
    ensure the result is bigger than zero (FEniCS
    takes care of tha too).
    '''
    def __init__( self, container ):
        
        # The container variable holds all required data,
        # parameters etc. See the documentation in module
        # container.py
        self.container = container
                
        # Create the expressions for the integrals of the 
        # enumrator(s) and denominator.
        self.enum0 = IntegratedExpression( container, "enum0_3d" )
        self.enum1 = IntegratedExpression( container, "enum1_3d" )
        self.enum2 = IntegratedExpression( container, "enum2_3d" )
        self.denom = IntegratedExpression( container, "denom_3d" )

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

        # Read file 
        loc_file = open( "cpp/" + expression_string + ".cpp" , 'r' )  
        
        # Extract code in string format
        code = loc_file.read()

        # Close file!!!
        loc_file.close()
        
        xp = Expression( code, degree = 4 ) 
        xp.kappa  = container.kappa 
        xp.factor = container.factor
        self.xp = xp

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
                self.xp.y[i] = y[i]

            # The return value is the an approximation to the integral of the expression
            value[0] = assemble( 
                self.xp * dx(self.container.mesh_obj)
                )

            # Keep track of what we calculated, so we don't have to do it again.
            self.dic[ y.tostring() ] = value

class Beta2DRadial(Expression):
    
    def __init__( self, container ):
        
        #self._degree = 2
        self._dim = container.dim
        
        # The container variable holds all required data,
        # parameters etc. See the documentation in module
        # container.py
        self.container = container
                
        self.G1 = radial.Radial( container, 1 )
        self.G2 = radial.Radial( container, 2 )
        
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
            self.G1.y = y
            self.G2.y = y
            G1 = interpolate( self.G1, self.container.V )
            G2 = interpolate( self.G2, self.container.V )
            mesh_obj = self.container.mesh_obj

            denom = 2 * assemble( G1 * G2     * dx(mesh_obj) )
            for i in range( self._dim ):
                enum  = assemble( (
                        grad(G2)[i]*G1 +
                        grad(G1)[i]*G2 
                        ) * dx(mesh_obj) )
                value[i]  = enum / denom
            
            # Keep track of what we calculated, so we don't have to do it again.
            self.dic[y.tostring()] = value


    def value_shape(self):
        return (2,)


class Beta3DRadial(Expression):
    
    def __init__( self, container ):
        
        self._degree = 2
        self._dim = container.dim
        
        # The container variable holds all required data,
        # parameters etc. See the documentation in module
        # container.py
        self.container = container
                
        self.G1 = radial.Radial( container, 1 )
        self.G2 = radial.Radial( container, 2 )
        
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
            self.G1.y = y
            self.G2.y = y
            G1 = interpolate( self.G1, self.container.V )
            G2 = interpolate( self.G2, self.container.V )
            mesh_obj = self.container.mesh_obj

            denom = 2 * assemble( G1 * G2     * dx(mesh_obj) )
            for i in range( self._dim ):
                enum  = assemble( (
                        grad(G2)[i]*G1 +
                        grad(G1)[i]*G2 
                        ) * dx(mesh_obj) )
                value[i]  = enum / denom
            
            # Keep track of what we calculated, so we don't have to do it again.
            self.dic[y.tostring()] = value


    def value_shape(self):
        return (3,)
