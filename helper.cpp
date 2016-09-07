#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <boost/math/special_functions/bessel.hpp>

#include "header.h"

using boost::math::cyl_bessel_k;
using namespace std;


int betaCube( unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval) {
  
  double kappa, *y;
  unpack( fdata, &kappa, &y );
  
  // distance from y
  double ra  = sqrt( (x[0]-y[0])*(x[0]-y[0]) + 
  		     (x[1]-y[1])*(x[1]-y[1]) +
  		     (x[2]-y[2])*(x[2]-y[2]) ) + 1E-9;
  
  double Khalf = cyl_bessel_k( 0.5 , kappa*ra );
  double expon = exp( -kappa*ra );
  
  double tot = kappa * Khalf * expon * ( 2 + 1/(kappa*ra) ) * pow( ra, -1.5 );
      
  fval[0] = tot * (y[0] - x[0]);
  fval[1] = tot * (y[1] - x[1]);
  fval[2] = tot * (y[2] - x[2]);
  fval[3] = 2.0 * Khalf * expon / sqrt( ra );

  return 0; 

}
  
int beta3D( unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval) {
  
  double kappa, det, *y, *vertices;
  unpack( fdata, &kappa, &det, &y, &vertices, ndim );
  
  double u[3];
  swap3D( x, u );
  
  A3D( u, vertices );
  
  double ra  = sqrt( (u[0]-y[0])*(u[0]-y[0]) + 
		     (u[1]-y[1])*(u[1]-y[1]) +
		     (u[2]-y[2])*(u[2]-y[2]) ) + 1E-9;
  
  double Khalf = cyl_bessel_k( 0.5 , kappa*ra );
  double expon = exp( -kappa*ra );
  
  double tot = kappa * Khalf * expon * ( 2 + 1/(kappa*ra) ) * pow( ra, -1.5 );
      
  fval[0] = det * tot * (y[0] - u[0]);
  fval[1] = det * tot * (y[1] - u[1]);
  fval[2] = det * tot * (y[2] - u[2]);
  fval[3] = det * 2.0 * Khalf * expon / sqrt( ra );

  return 0; 

}

int simple3D( unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval) {
  
  double kappa, det, *y, *vertices;
  unpack( fdata, &kappa, &det, &y, &vertices, ndim );
  
  double u[3];
  swap3D( x, u );
      
  A3D( u, vertices );
  
  fval[0] = (1./6) * det * u[0];
  
  return 0;
}

int constant3D( unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval) {
    
  double kappa, det, *y, *vertices;
  unpack( fdata, &kappa, &det, &y, &vertices, ndim );

  // 12 because we break the unit cube into six pyramids (one
  // for every facet), map to a representative pyramid and then
  // break it in half.
  fval[0] = det / 12;
  
  return 0;
}

void A3D ( double *v, const double * vertices ) {
  
  const double * a = vertices;
  const double * b = vertices + 3;
  const double * c = vertices + 6;
  const double * d = vertices + 9;
  
  double tmp[3];
  tmp[0] = a[0] + (2 * d[0] - c[0] - a[0]) * v[0] + (b[0] - a[0]) * v[1] + (c[0] - b[0]) * v[2];
  tmp[1] = a[1] + (2 * d[1] - c[1] - a[1]) * v[0] + (b[1] - a[1]) * v[1] + (c[1] - b[1]) * v[2];
  tmp[2] = a[2] + (2 * d[2] - c[2] - a[2]) * v[0] + (b[2] - a[2]) * v[1] + (c[2] - b[2]) * v[2];
    
  v[0] = tmp[0];
  v[1] = tmp[1];
  v[2] = tmp[2];
}

double dist3D( const double * x, const double * y ) {
  
  return sqrt( (x[0]-y[0])*(x[0]-y[0]) +
	       (x[1]-y[1])*(x[1]-y[1]) +
	       (x[2]-y[2])*(x[2]-y[2]) 
	       );
}

int closest( const double * x ) { 
  
  
  const double pointsOnCube[18] =
    { 0  , 0.5, 0.5,
      1  , 0.5, 0.5,
      0.5, 0  , 0.5,
      0.5, 1  , 0.5,
      0.5, 0.5, 0  ,
      0.5, 0.5, 1   };
  
  // Find in which "pyramid" x is located
  double minDist = 10;
  double currDist;
  int minIndex;
  
  for (int i = 0 ; i < 6 ; i++ ) {
    currDist = dist3D( pointsOnCube + 3*i, x );
    if ( currDist < minDist ) {
      minDist = currDist;
      minIndex = i;
    }
  }
  return minIndex;
}

void swap3D( const double *x, double * y ) {
    
  
  int minIndex = closest( x ); 
  switch ( minIndex ) {
    
  case 0:
    memcpy( y, x, 3*sizeof(double) );
    break;
  case 1:
    y[0] = 1 - x[0];
    y[1] = x[1];
    y[2] = x[2];
    break;
    
  case 2:
    y[0] = x[1];
    y[1] = x[0];
    y[2] = x[2];
    break;
  case 3:
    y[0] = 1 - x[1];
    y[1] = x[0];
    y[2] = x[2];
    break;
    
  case 4:
    y[0] = x[2];
    y[1] = x[1];
    y[2] = x[0];
    break;
  case 5:
    y[0] = 1 - x[2];
    y[1] = x[1];
    y[2] = x[0];
    break;
    
  
  }  
  
  if (y[1] < y[2]) {
    double tmp = y[1];
    y[1] = y[2];
    y[2] = tmp;
  }
}


int beta2D( unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval) {
  
  double kappa, det, *y, *vertices;
  unpack( fdata, &kappa, &det, &y, &vertices, ndim );
  
  double u[2];
  swap2D( x, u );
  
  A2D( u, vertices );
  
  double ra  = sqrt( (u[0]-y[0])*(u[0]-y[0])  +  (u[1]-y[1])*(u[1]-y[1]) ) + 1E-9;
  
  double phi0 = cyl_bessel_k( 0.0, kappa*ra );
  double phi1 = cyl_bessel_k( 1.0, kappa*ra );
  double tot  = kappa * (phi0*phi0 + phi1*phi1);
  
  fval[0] = 0.5 * det * tot * (y[0] - u[0]); // Enumerator[0]
  fval[1] = 0.5 * det * tot * (y[1] - u[1]); // Enumerator[1]
  fval[2] =       det * ra  *  phi0 * phi1 ; // Denomintaor

  return 0; 

}

int simple2D(unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval) {

  double kappa, det, *y, *vertices;
  unpack( fdata, &kappa, &det, &y, &vertices, ndim );
  //printData( kappa, det, y, vertices, ndim );
    
  double u[2];
  swap2D( x, u );
  
  A2D( u, vertices );
 
  // Multiply by the determinant at its designated spot
  fval[0] =  0.5 * det * u[0];
  
  return 0; 
}

int singular2D(unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval) {

  double kappa, det, *y, *vertices;
  unpack( fdata, &kappa, &det, &y, &vertices, ndim );

  double u[2];
  swap2D( x, u );
  
  A2D( u, vertices );
  
  fval[0] = - 0.5 * det * log( sqrt(u[0]*u[0] + u[1]*u[1]) );
  
  return 0; 

}

double detA ( const double * v, int n ) {
  /*
    Divide the absolute value of the determinant by 2 since we integrate
    over the unit square, whereas whe should be integrating over half
    of it
  */
  if ( n == 2 )  
    return fabs
      (	(v[2] - v[0]) * (v[5] - v[3]) - 
	(v[4] - v[2]) * (v[3] - v[1]) 
	);
  else {
    const double * a = v;
    const double * b = v + 3;
    const double * c = v + 6;
    const double * d = v + 9;
    
    return fabs
      (
       (2*d[0]-c[0]-a[0]) * (b[1]-a[1]) * (c[2]-b[2]) + 
       (2*d[1]-c[1]-a[1]) * (b[2]-a[2]) * (c[0]-b[0]) +
       (2*d[2]-c[2]-a[2]) * (b[0]-a[0]) * (c[1]-b[1]) -
       (2*d[2]-c[2]-a[2]) * (b[1]-a[1]) * (c[0]-b[0]) -
       (2*d[1]-c[1]-a[1]) * (b[0]-a[0]) * (c[2]-b[2]) -
       (2*d[0]-c[0]-a[0]) * (b[2]-a[2]) * (c[1]-b[1])
       );
  }
}


void A2D ( double *v, const double * vertices ) {
  
  const double * a = vertices;
  const double * b = vertices + 2;
  const double * c = vertices + 4;
  
  double tmp[2];
  tmp[0] = a[0] + (b[0] - a[0]) * v[0] + (c[0] - b[0]) * v[1]; 
  tmp[1] = a[1] + (b[1] - a[1]) * v[0] + (c[1] - b[1]) * v[1];
  
  v[0] = tmp[0];
  v[1] = tmp[1];
}



void swap2D( const double *x, double * y ) {
  
  if (x[0] > x[1]) {
    
    y[0] = x[0];
    y[1] = x[1];
  
  } else {
  
    y[0] = x[1];
    y[1] = x[0];
  
  }
   
}

void unpack ( void * fdata, 
	      double * kappa, 
	      double * det,
	      double ** y,
	      double ** vertices,
	      int ndim )
{
  *kappa    = ((double *) fdata )[0];
  *det      = ((double *) fdata )[1];
  *y        =  (double *) fdata + 2;
  *vertices =  (double *) fdata + 2 + ndim;
}

void unpack ( void * fdata, 
	      double * kappa, 
	      double ** y )
{
  *kappa = ((double *) fdata )[0];
  *y     =  (double *) fdata + 1;
}

void printData( double kappa, double det, double * y, const double * vertices, int ndim ) {
  
  cout << "Kappa   = " << kappa << "\n";
  cout << "Det     = " << det   << "\n";
  cout << "y = \n"; 
  printArray( ndim, 1, y );
  cout << "vertices = \n";
  printArray( ndim + 1, ndim, vertices );
}

void printData( double * fdata, int ndim ) {
  cout << "Kappa   = " << fdata[0] << "\n";
  cout << "Det     = " << fdata[1]   << "\n";
  cout << "y = \n"; 
  printArray( 1, ndim, fdata + 2 );
  cout << "vertices = \n";
  printArray( ndim + 1, ndim, fdata + 2 + ndim );
}

void printArray( int lines, int columns, double * a ) {
    
  for( int i = 0 ; i < lines ; i ++ ) {
    for( int j = 0 ; j < columns ; j ++ ) {
      cout << a[ i*columns + j ];
      if ( j < columns - 1 )
	cout << " , ";
    }
    cout << "\n";
  }  
}

void printArray( int lines, int columns, const double * a ) {
    
  for( int i = 0 ; i < lines ; i ++ ) {
    for( int j = 0 ; j < columns ; j ++ ) {
      cout << a[ i*columns + j ];
      if ( j < columns - 1 )
	cout << " , ";
    }
    cout << "\n";
  }  
}

