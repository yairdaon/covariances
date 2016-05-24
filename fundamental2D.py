from dolfin import *
import numpy as np
import helper

#########################################################
# Fundamental solution ##################################
#########################################################
def fundamental( container, mode ):
    fund   = Function( container.V )
    fund_xpr = container.generate( "fundamental" )
    pt = helper.get_source( container.mesh_name )
    helper.update_x_xp( pt, fund_xpr )
    fund = Function( container.V )
    fund.interpolate( fund_xpr )
    helper.save_plots( fund, 
                       ["Free Space", "Greens Function"], 
                       container )


