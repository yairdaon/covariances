#include <math.h>
#include <stdlib.h> 
#include <stdio.h>
#include <iostream>

#include "cubature.h"
#include "header.h"

using namespace std;
/*
  This file integrates the function
  betaCube centered at a specific location 
  y on the unit cube. The four components
  of the function are three components of the
  enumerator and one of the denominator.
 */



void integrateVectorOnCube( int n,
			    double * y, 
			    double kappa, 
			    double tol,
			    int fdim,
			    double * result ) {
  double err[4];
  double xmin[3] = {0,0,0};
  double xmax[3] = {1,1,1};
      
  double fdata[1 + 3];
  
  fdata[0] = kappa;
  memcpy( fdata + 1, y, 3*sizeof(double) );
  
  hcubature( 4,                //unsigned fdim
	     betaCube,         //integrand f - need to replace this line, ow won't compile!!
	     fdata,            //void *fdata
	     3,                //unsigned dim
	     xmin,             //const double *xmin
	     xmax,             //const double *xmax 
	     0,                //size_t maxEval
	     tol,              //double reqAbsError 
	     tol,              //double reqRelError 
	     ERROR_INDIVIDUAL, //error_norm norm
	     result,          //double *val
	     err );            //double *err
	      
}
