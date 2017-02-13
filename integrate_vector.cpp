#include <math.h>
#include <stdlib.h> 
#include <stdio.h>

#include "header.h"
#include "cubature.h"

void integrateVector( int n,
		      double * y, 
		      double kappa, 
		      int m, 
		      double * vertices, 
		      double tol,
		      int fdim,
		      double * result ) {
  /*
    Integrates and adds an n-dimensional vector valued function.
    The idea you should have in mind is performing Riemann integration
    over cells. Then you add the result of every cell to the overall
    sum.
  */
  
  double * val  = new double[fdim];
  double * err  = new double[fdim];
  double * xmin = new double[n   ];
  double * xmax = new double[n   ];
  
  // Vectors live inside the n-dimensional cube. Then
  // we stretch, translate and rotate to get to each cell.
  for (int i = 0; i < n ; i++ ) {
    xmin[i] = 0;
    xmax[i] = 1;
  }
  
  double * fdata = new double[1 + 1 + n + m];
  
  fdata[0] = kappa;
  fdata[1] = detA( vertices, n );
  memcpy( fdata + 2,     y,        n * sizeof(double) );
  memcpy( fdata + 2 + n, vertices, m * sizeof(double) );
  
  hcubature( fdim,             //unsigned fdim
	     FUNCTION_NAME,    //integrand f - need to replace this line, ow won't compile!!
	     fdata,            //void *fdata
	     n,                //unsigned dim
	     xmin,             //const double *xmin
	     xmax,             //const double *xmax 
	     0,                //size_t maxEval
	     tol,              //double reqAbsError 
	     tol,              //double reqRelError 
	     ERROR_INDIVIDUAL, //error_norm norm
	     val,              //double *valx
	     err );            //double *err
	      
 
  for (int i = 0; i < fdim ; i++ ) {
    result[i] = result[i] + val[i];

    // Catch NaNs, as a precaution let's say
    if isnan( val[i] ) {
	printf( "NaN! Vertices = \n" );
	printData( fdata, 2 );
      }
  }

  delete[] fdata;
  delete[] val;
  delete[] err;
  delete[] xmin;
  delete[] xmax;
}

