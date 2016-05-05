from dolfin import *
import numpy as np
import helper

#########################################################
# Fundamental solution ##################################
#########################################################
def fundamental( container, mode ):
    fund   = Function( container.V )
    fund_xpr = container.generate( "fundamental" )
    pt = helper.pts[container.mesh_name][0]
    helper.update_x_xp( pt, fund_xpr )
    fund = Function( container.V )
    fund.interpolate( fund_xpr )
    print np.max( fund.vector().array() )
    print container.sig2
    helper.save_plots( fund, 
                       ["Free Space", "Greens Function"], 
                       container.mesh_name,
                       ran = container.ran_sol,
                       mode = mode,
                       scalarbar = True )


