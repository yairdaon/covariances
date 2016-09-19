import numpy as np
import hashlib 
import math
from scipy.special import kn as kn
from scipy.special import kv as kv

import dolfin as dol
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
        self.strictEval( value, y )
        if self.dim == 2:
            # self.fullDic [ (y[0], y[1]) ] = evalTup 
            self.tupDic[ (y[0], y[1]) ] = (value[0], value[1]) 
        elif self.dim == 3:
            # self.fullDic [ (y[0], y[1], y[2]) ] = evalTup 
            self.tupDic[ (y[0], y[1], y[2]) ] = (value[0], value[1], value[2]) 
        else:
            raise ValueError( "Variable dim must equal 2 or 3" )

def generalInit( self, cot ):
        
    self.V = dol.FunctionSpace( cot.mesh_obj, "DG", 0 )
    u = dol.TrialFunction( self.V )
    v = dol.TestFunction ( self.V )
    self.M = dol.assemble( u*v*dol.dx )

    self.enum0 = dol.Function( self.V )
    self.enum1 = dol.Function( self.V )
    if cot.dim == 3:
        self.enum2 = dol.Function( self.V )
    self.denom = dol.Function( self.V )

    # Get all coordiantes
    self.x = self.V.tabulate_dof_coordinates()
    self.x.resize(( self.V.dim(), cot.dim ))
    
    self.cot = cot #See the documentation in cotontainer.py
    self.dim = cot.dim
    self.valDic = {}
    self.tupDic = {}
    self.fullDic = {}

def getRaKappara( self, y ):
    
    y_x = y - self.x
    ra  = np.sqrt( np.sum( y_x * y_x, axis = 1 ) )
    kappara = self.cot.kappa * ra

    return ra, kappara, y_x
      

#################################################################
# Finite Element betas using FEniCS package #####################
################################################################# 
class Beta2D(dol.Expression):
 
    def __init__( self, cot ):
  
        # For some reason it is impossible to subclass
        # an Expression so I use this instead.
        generalInit( self , cot )
  
    eval = generalEval
    def strictEval( self, value, y ):
            
        ra, kappara, y_x = getRaKappara(self, y )
        
        bess0 = kn( 0, kappara )
        bess1 = kn( 1, kappara )
        enums = self.cot.kappa * ( bess0*bess0 + bess1*bess1 ) 

        enum  = np.einsum( "i , ik -> ik", enums, y_x )
        denom = 2 * ra * bess0 * bess1   
        
        self.enum0.vector().set_local( enum[:,0] )
        self.enum1.vector().set_local( enum[:,1] )
        self.denom.vector().set_local( denom     )

        self.enum0.vector().set_local( (self.M * self.enum0.vector()).array() )
        self.enum1.vector().set_local( (self.M * self.enum1.vector()).array() )
        self.denom.vector().set_local( (self.M * self.denom.vector()).array() )
        
        denom = np.sum( self.denom.vector().array() )
    
        value[0] = np.sum( self.enum0.vector().array() ) / denom
        value[1] = np.sum( self.enum1.vector().array() ) / denom
  
      
    def value_shape(self):
        return (2,)     
       

class Beta3D(dol.Expression):

    def __init__( self, cot ):
        
        # For some rason it is impossible to subclass
        # an Expression so I use this instead.
        generalInit( self, cot )
        
    eval = generalEval
    def strictEval( self, value, y ):
        
        ra, kappara, y_x = getRaKappara(self, y )
        
        expon = np.exp(-kappara) 
        bess  = kv( 0.5, kappara )
        enums = self.cot.kappa * np.power(ra,-1.5) * (2+1./kappara) * expon * bess 

        enum  = np.einsum( "i , ik -> ik", enums, y_x )
        denom = 2 * bess * expon * np.power(ra,-0.5) 
        
        self.enum0.vector().set_local( enum[:,0] )
        self.enum1.vector().set_local( enum[:,1] )
        self.enum2.vector().set_local( enum[:,2] )
        self.denom.vector().set_local( denom            )

        self.enum0.vector().set_local( (self.M * self.enum0.vector()).array() )
        self.enum1.vector().set_local( (self.M * self.enum1.vector()).array() )
        self.enum2.vector().set_local( (self.M * self.enum2.vector()).array() )
        self.denom.vector().set_local( (self.M * self.denom.vector()).array() )
        
        denom = np.sum( self.denom.vector().array() )
    
        value[0] = np.sum( self.enum0.vector().array() ) / denom
        value[1] = np.sum( self.enum1.vector().array() ) / denom
        value[2] = np.sum( self.enum2.vector().array() ) / denom
   
    def value_shape(self):
        return (3,)


#################################################################
# Adaptive betas using the cubature package #####################
#################################################################
class Beta2DAdaptive(dol.Expression):
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
        self.elements = dict((cell.index(), cell.entities(0)) for cell in dol.cells(cot.mesh_obj))
    
        self.tol = tol
        
    eval = generalEval
    def strictEval( self, value, y ):
        
        # the dimension
        d = self.dim
    
        result = np.zeros( (d+1,) )
        vertices = np.empty(  (d+1)*d  )
     
        for cell in self.cot.mesh_obj.cells():
            i = 0
            for vertex in cell:
                vertices[i:i+d] = self.coordinates[vertex]
                i = i + d
            if np.max( vertices ) > 1e10:
                print "Bad vertex!!!"
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
             
class BetaCubeAdaptive(dol.Expression):
        
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
        
class Beta2DRadial(dol.Expression):
    
    def __init__( self, cot ):
        
        generalInit( self, cot )
                
        self.G1, self.dG1, self.G2, self.dG2 = radial.radial(cot)
                  
    eval = generalEval
    def strictEval( self, value, y ):
        
        ra, kappara, y_x = getRaKappara(self, y )
        
        ens = -( # Minus sign because does NOT cancel out here!
            self.G1( ra ) * self.dG2( ra ) +
            self.G2( ra ) * self.dG1( ra ) 
            )
        en  = np.einsum( "i , ik -> ik", ens/ra, y_x )
        
        self.enum0.vector().set_local( en[:,0] )
        self.enum0.vector().set_local( (self.M * self.enum0.vector()).array() )
        
        self.enum1.vector().set_local( en[:,1] )
        self.enum1.vector().set_local( (self.M * self.enum1.vector()).array() )
    
        # Note that we DO NOT muply by kappa, since the derivatives
        # dG1 and dG2 are wrt kappa * r !!!
        en0 = np.sum( self.enum0.vector().array() )
        en1 = np.sum( self.enum1.vector().array() )

        
        den = self.G2( ra ) * self.G1( ra )    
        self.denom.vector().set_local( den     )
        self.denom.vector().set_local( (self.M * self.denom.vector()).array() )
        den = 2 * np.sum( self.denom.vector().array() )
        
        
        value[0] = en0 / den
        value[1] = en1 / den
     
    def value_shape(self):
        return (2,)

class Beta3DRadial(dol.Expression):
    
    def __init__( self, cot ):
        
        generalInit( self, cot )
                
        self.G1, self.dG1, self.G2, self.dG2 = radial.radial(cot)
                  
    eval = generalEval
    def strictEval( self, value, y ):
        
        ra, kappara, y_x = getRaKappara(self, y )
        
        ens = -( 
            self.G1( ra ) * self.dG2( ra ) +
            self.G2( ra ) * self.dG1( ra )
            )
        en  = np.einsum( "i , ik -> ik", ens/ra, y_x )
        
        self.enum0.vector().set_local( en[:,0] )
        self.enum0.vector().set_local( (self.M * self.enum0.vector()).array() )
        
        self.enum1.vector().set_local( en[:,1] )
        self.enum1.vector().set_local( (self.M * self.enum1.vector()).array() )
    
        self.enum2.vector().set_local( en[:,2] )
        self.enum2.vector().set_local( (self.M * self.enum2.vector()).array() )
    
        en0 = np.sum( self.enum0.vector().array() )
        en1 = np.sum( self.enum1.vector().array() )
        en2 = np.sum( self.enum2.vector().array() )

        
        den = self.G2( ra ) * self.G1( ra )    
        self.denom.vector().set_local( den     )
        self.denom.vector().set_local( (self.M * self.denom.vector()).array() )
        den = 2 * np.sum( self.denom.vector().array() )
        
        
        value[0] = en0 / den
        value[1] = en1 / den
        value[2] = en2 / den
     
    def value_shape(self):
        return (3,)


class IntegratedExpression(dol.Expression):
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
        
        xp = dol.Expression( code, degree = 2 ) 
        xp.kappa  = cot.kappa 
        xp.factor = cot.factor
        self.xp = xp

        self.V = dol.FunctionSpace( cot.mesh_obj, "DG", 0 )
        u = dol.TrialFunction( self.V )
        v = dol.TestFunction ( self.V )
        self.M = dol.assemble( u*v*dol.dx )
        
        # tmp = Function( V )
        # tmp.vector().set_local( np.ones( tmp.vector().array().shape ) )
        # self.ones = tmp


    def eval( self, value, y ):
        
        # Let the expression know with respect to which (boundary)
        # point we want its value.
        for i in range( self.cot.dim ):
            self.xp.y[i] = y[i]
        # import time
      
        # start = time.time()            
        
        #tmp = dol.interpolate( self.xp, self.V )
        
        # print "Elapsed " + str( time.time() - start )
        # tmp = self.M * tmp.vector()
        
        # The return value is the an approximation to the integral of the expression
        # value[0] = np.sum( tmp.array() )
        value[0] = dol.assemble( self.xp * dol.dx(self.cot.mesh_obj) )


