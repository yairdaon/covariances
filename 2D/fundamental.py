from dolfin import *
import numpy as np
import helper

#########################################################
# Fundamental solution ##################################
#########################################################
def fundamental( container, mode ):
    fund   = Function( container.V )
    fund_xpr = container.generate( "mat12" )
    pt = helper.pts[container.mesh_name][0]
    fund_xpr.x[0] = pt[0]
    fund_xpr.x[1] = pt[1]
    fund = Function( container.V )
    fund.interpolate( fund_xpr )
    fund.vector()[:] = -fund.vector()[:]
    helper.save_plots( fund, 
                       "Fundamental Solution", 
                       container.mesh_name,
                       ran = container.ran_sol,
                       mode )


