import numpy as np
import hashlib 
import math

from dolfin import *
import instant

import helper
import radial


#################################################################
# Used since you can't sub-subclass an Expression ###############
#################################################################

def generalEval( self, value, y ):
        
    # If we already did the calculation, return the value...
    if self.valDic.has_key( y.tostring() ):
        value = self.dic[ y.tostring() ]
    else:
        evalTup = self.strictEval( value, y )
        if self.dim == 2:
            self.fullDic [ (y[0], y[1]) ] = evalTup 
            self.tupDic[ (y[0], y[1]) ] = (value[0], value[1]) 
        elif self.dim == 3:
            self.fullDic [ (y[0], y[1], y[2]) ] = evalTup 
            self.tupDic[ (y[0], y[1], y[2]) ] = (value[0], value[1], value[2]) 
        else:
            raise ValueError( "Variable dim must equal 2 or 3" )

def generalInit( self, cot ):
        
    self.cot = cot #See the documentation in cotontainer.py
    self.dim = cot.dim
    self.valDic = {}
    self.tupDic = {}
    self.fullDic = {}


#################################################################
# Finite Element betas using FEniCS package #####################
################################################################# 
class Beta2D(Expression):
 
    def __init__( self, cot ):
  
        # For some rason it is impossible to subclass
        # an Expression so I use this instead.
        generalInit( self , cot )
  
        # Create the expressions for the integrals of the 
        # enumrator(s) and denominator.
        self.enum0 = IntegratedExpression( cot, "enum0_2d" )
        self.enum1 = IntegratedExpression( cot, "enum1_2d" )
        self.denom = IntegratedExpression( cot, "denom_2d" )
     
    eval = generalEval
    def strictEval( self, value, y ):
              
        # FEniCS takes an inner product with the unit
        # normal to obtain our beta.
        denom_variable = self.denom( y )
        value[0]  = self.enum0( y ) / denom_variable
        value[1]  = self.enum1( y ) / denom_variable
    
    def value_shape(self):
        return (2,)     
       

class Beta3D(Expression):

    def __init__( self, cot ):
        
        # For some rason it is impossible to subclass
        # an Expression so I use this instead.
        generalInit( self, cot )
                
        # Create the expressions for the integrals of the 
        # enumrator(s) and denominator.
        self.enum0 = IntegratedExpression( cot, "enum0_3d" )
        self.enum1 = IntegratedExpression( cot, "enum1_3d" )
        self.enum2 = IntegratedExpression( cot, "enum2_3d" )
        self.denom = IntegratedExpression( cot, "denom_3d" )

    eval = generalEval
    def strictEval( self, value, y ):

        # FEniCS takes an inner product with the unit
        # normal to obtain our beta.
        denom_variable = self.denom( y )
        en0 = self.enum0( y )
        en1 = self.enum1( y )
        en2 = self.enum2( y )
        value[0]  = en0 / denom_variable
        value[1]  = en1 / denom_variable
        value[2]  = en2 / denom_variable
        
        return ( en0, en1, en2, denom_variable )

    def value_shape(self):
        return (3,)


class IntegratedExpression(Expression):
    '''
    Compile and integrate a given expression. Expression
    is given as a string that refers to a c++ file
    which is then compiled.
    '''
    
    def __init__( self, cot, expression_string ):
                
        self.cot = cot # See the documentation in container.py
        
        # Generate the expression from the c++ file that
        # the input variable expresion_string refers to.

        # Read file 
        loc_file = open( "cpp/" + expression_string + ".cpp" , 'r' )  
        
        # Extract code in string format
        code = loc_file.read()

        # Close file!!!
        loc_file.close()
        
        xp = Expression( code, degree = 2 ) 
        xp.kappa  = cot.kappa 
        xp.factor = cot.factor
        self.xp = xp

    def eval( self, value, y ):
        
        # Let the expression know with respect to which (boundary)
        # point we want its value.
        for i in range( self.cot.dim ):
            self.xp.y[i] = y[i]
            
        # The return value is the an approximation to the integral of the expression
        value[0] = assemble( self.xp * dx(self.cot.mesh_obj) )





#################################################################
# Adaptive betas using the cubature package #####################
#################################################################
class Beta2DAdaptive(Expression):
    '''
    Compile and integrate a given expression. Expression
    is given as a string that refers to a c++ file
    which is then compiled.
    '''
    
    def __init__( self, cot, tol = 1e-9 ):
        
        generalInit( self , cot )
        
        # Generate the expression from the c++ file that
        # the input variable expresion_string refers to.
        self.xpr = generateInstant( "beta2D" )
            
        self.coordinates = cot.mesh_obj.coordinates()
        self.elements = dict((cell.index(), cell.entities(0)) for cell in cells(cot.mesh_obj))
    
        self.tol = tol
        
    eval = generalEval
    def strictEval( self, value, y ):
        
        # the dimension
        d = self.dim
    
        result = np.zeros( (d+1,) )
        vertices = np.empty(  (d+1)*d  )
     
        for element in range(self.cot.mesh_obj.num_cells()):
            i = 0
            for vertex in self.elements[element]:
                vertices[i:i+d] = self.coordinates[vertex]
                i = i + d
                self.xpr.integrateVector(  y, self.cot.kappa, vertices, self.tol, result )  
                
        # The last entry is the denominator, by which we divide.
        # The first entries are the enumerators, which is a vector
        # that will later be multiplied by a unit normal
        for i in range( d ):
            value[i] = result[i] / result[d]
        
    
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
        
    def __init__( self, cot, tol = 1e-9 ):
        
        generalInit( self, cot )

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
                
        self.tol = tol
        
    eval = generalEval
    def strictEval( self, value, y ):
        result = np.empty( 4 )
        
        self.xpr.integrateVectorOnCube( y, 
                                        self.cot.kappa, 
                                        self.tol, 
                                        result )  
            
        # The last entry is the denominator, by which we divide.
        # The first entries are the enumerators, which is a vector
        # that will later be multiplied by a unit normal
        for i in range( 3 ):
            value[i] = result[i] / result[3]

    def value_shape(self):
        return (3,)
    



#################################################################
# Radial betas that Stadler forced me to code ###################
#################################################################
        
class Beta2DRadial(Expression):
    
    def __init__( self, cot ):
        
        generalInit( self, cot )
                
        self.G1 = radial.Radial( cot, 1 )
        self.G2 = radial.Radial( cot, 2 )
                  
    eval = generalEval
    def strictEval( self, value, y ):
        
        self.G1.y = y
        self.G2.y = y
        G1 = interpolate( self.G1, self.cot.V )
        G2 = interpolate( self.G2, self.cot.V )
        mesh_obj = self.cot.mesh_obj

        denom = 2 * assemble( G1 * G2     * dx(mesh_obj) )
        for i in range( self._dim ):
            enum  = assemble( (
                grad(G2)[i]*G1 +
                grad(G1)[i]*G2 
            ) * dx(mesh_obj) )
            value[i]  = enum / denom

    def value_shape(self):
        return (2,)


class Beta3DRadial(Expression):
    
    def __init__( self, cot ):

        generalInit( self, cot )
                
        self.G1 = radial.Radial( cot, 1 )
        self.G2 = radial.Radial( cot, 2 )
        
    eval = generalEval
    def strictEval( self, value, y ):
        
        self.G1.y = y
        self.G2.y = y
        G1 = interpolate( self.G1, self.cot.V )
        G2 = interpolate( self.G2, self.cot.V )
        mesh_obj = self.cot.mesh_obj

        denom = 2 * assemble( G1 * G2     * dx(mesh_obj) )
        for i in range( self._dim ):
            enum  = assemble( (
                grad(G2)[i]*G1 +
                grad(G1)[i]*G2 
            ) * dx(mesh_obj) )
            value[i]  = enum / denom
        
    def value_shape(self):
        return (3,)

