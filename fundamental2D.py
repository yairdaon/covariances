from dolfin import *
import numpy as np
import helper
import betas2D

#########################################################
# Fundamental solution ##################################
#########################################################
def fundamental( container, mode ):
    fund   = Function( container.V )
    fund_xpr = container.generate( "fundamental" )
    pt = helper.get_source( container.mesh_name )
    fund_xpr.y[0] = pt[0]
    fund_xpr.y[1] = pt[1]
    fund = Function( container.V )
    fund.interpolate( fund_xpr )
    helper.save_plots( fund, 
                       ["Free Space", "Greens Function"], 
                       container )


