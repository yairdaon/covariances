from dolfin import *
from scipy import special as sp
import numpy as np
import helper


kappa = 1.27
x  = np.array( [1.0 , 1.0 ] )
y  = np.array( [0.0 , 1.0 ] )
yn = np.array( [1.24, 2.3 ] )

G, dGdn = helper.init_expr( kappa )
helper.update( G, dGdn, y, yn )

jit = G( x )
pyt = sp.kn(0, kappa*1.0 )
print jit
print pyt

assert abs( jit - pyt ) < 1E-12

######
jit = dGdn( x )
diff = x - y
r = np.linalg.norm( diff )
pyt = -kappa * sp.kn( 1 , kappa * r ) * np.einsum( "i,i", diff, yn ) / r
print jit
print pyt

assert abs( jit - pyt ) < DOLFIN_EPS
