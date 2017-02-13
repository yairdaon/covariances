#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cubature.h"
#include "header.h"


using namespace std;

/*
  Tests!! Run them and see everything works.
 */
void testSimple2D() {
 
  double fdata[1 + 1 + 2 + 6];
  double xmin[2] = {0,0}, xmax[2] = {1,1}, val, err, tol = 1e-5;
  
  
  const double verticesExam[6] = {
    -3, 0, // a
    6,  0, // b
    0,  3 // c
  };
 
  cout << "Det(A) = " <<  detA( verticesExam, 2 ) << "\n";;
  
  double v[2] = { 1, 1 }; // Initial values
 
  A2D( v, verticesExam );
  cout << "Av = \n";
  printArray( 2, 1, v );
  

  fdata[1] = detA( verticesExam, 2 );
  memcpy( fdata + 4 , verticesExam, 6 * sizeof(double) );
  
  hcubature(1, simple2D, &fdata, 2, xmin, xmax, 0, 0, tol, ERROR_INDIVIDUAL, &val, &err);
  
  cout << "Example from https://www.math.washington.edu/~king/coursedir/m324a10/quiz/quiz4-answers.pdf\n";
  cout << "Computed integral = " << val << " +/- " << err << "\n";
  cout << "Analytic solution = " << 13.5                    << "\n";
  cout << "Difference        = " << val - 13.5            << "\n";
  cout << "Tolerance         = " << tol                     << "\n";
  
}

void testSingular2D() {

  double fdata[1 + 1 + 2 + 6];
  double xmin[2] = {0,0}, xmax[2] = {1,1}, val, err, tol = 1e-5;
  
  const double verticesSingular[6] = {
    0, 0, // a
    1, 0, // b
    1, 1 // c
  };

  fdata[1] = detA( verticesSingular, 2 );
  memcpy( fdata + 4 , verticesSingular, 6 * sizeof(double) );

  hcubature(1, singular2D, &fdata, 2, xmin, xmax, 0, 0, tol, ERROR_INDIVIDUAL, &val, &err);
  

  cout << "\n\nExample I made\n";
  cout << "Computed integral = " << val << " +/- " << err      << "\n";
  cout << "Numerical Matlab  = " <<         0.184014120655140  << "\n";
  cout << "Difference        = " << val - 0.184014120655140    << "\n";
  cout << "Tolerance         = " << tol                        << "\n";
   
}

void testSimpleA3D() {
  cout << "Starting testSimpleA3D\n";
  
  const double stdTet[12] = {
    0   , 0   , 0   , // a
    0   , 1   , 0   , // b
    0   , 1   , 1   , // c
    0.5 , 0.5 , 0.5   // d
  };
  
  const double vertices3D[12] = {
    0   , 0   , 0   , // a
    1   , 0   , 0   , // b
    0   , 1   , 0   , // c
    0   , 0   , 1     // d
  };
      
  double u[3];
  for (int i = 0 ; i < 4 ; i++ ) {
    
    memcpy( u, stdTet + i*3 , 3 *sizeof(double) );
    cout << "u    = ";
    printArray( 1, 3, u );
    cout << "Au   = ";
    A3D ( u, vertices3D );
    printArray( 1, 3, u );
    cout << "Test = ";
    printArray( 1, 3, vertices3D + 3*i );
    cout << "\n";
  }
  cout << "Finished testSimpleA3D\n";

}

void testHardA3D() {
  cout << "Starting testHardA3D\n";
  
  const double stdTet[12] = {
    0   , 0   , 0   , // a
    0   , 1   , 0   , // b
    0   , 1   , 1   , // c
    0.5 , 0.5 , 0.5   // d
  };

  const double vertices3D[12] = {
    4, -3, 1  , // a
    3, 0 , 8  , // b
    1, 2 , 9.2, // c
    4, 5 , 6  // d
  };

  double u[3];
  for (int i = 0 ; i < 4 ; i++ ) {
    
    memcpy( u, stdTet + i*3 , 3 * sizeof(double) );
    
    cout << "u    = ";
    printArray( 1, 3, u );
    cout << "Au   = ";
    A3D ( u, vertices3D );
    printArray( 1, 3, u );
    cout << "Test = ";
    printArray( 1, 3, vertices3D + 3*i );
    cout << "\n";
  }
  cout << "Finished testHardA3D\n";

}

void testSwap3D() {
 
  cout << "\nStarting testSwap3D";
  double y [3];
  
  double x3[3] = { 0.5, 0.5, 0 };
  swap3D( x3, y );
  cout << "\nSource      = ";
  printArray( 1, 3, x3 );
  cout << "Destination = ";
  printArray( 1, 3, y );
  
    
  double x4[3] = { 1, 0.5, 0.5 };
  swap3D( x4, y );
  cout << "\nSource      = ";
  printArray( 1, 3, x4 );
  cout << "Destination = ";
  printArray( 1, 3, y );
  
   
  
  double x5[3] = { 0.5, 1, 0.5 };
  swap3D( x5, y );
  cout << "\nSource      = ";
  printArray( 1, 3, x5 );
  cout << "Destination = ";
  printArray( 1, 3, y );
  
  double x6[3] = { 0.5, 0, 0.5 };
  swap3D( x6, y );
  cout << "\nSource      = ";
  printArray( 1, 3, x6 );
  cout << "Destination = ";
  printArray( 1, 3, y );
  
  
  
  
  cout << "Finished testSwap3D\n";
}

void testSimplexVolume3D() {
  
  double fdata[1 + 1 + 3 + 12];
  double xmin[3] = {0,0,0}, xmax[3] = {1,1,1}, val, err, tol = 1e-5;
  
  
  const double simpleVertices3D[12] = {
    0   , 0   , 0   , // a
    1   , 0   , 0   , // b
    0   , 1   , 0   , // c
    0   , 0   , 1     // d
  };
  
  const double stdTet[12] = {
    0   , 0   , 0   , // a
    0   , 1   , 0   , // b
    0   , 1   , 1   , // c
    0.5 , 0.5 , 0.5   // d
  };
  
      
  fdata[0] = 1.23;
  fdata[1] = detA( simpleVertices3D, 3 );
  fdata[2] = 1.1;// Ignoring y for now, in entries 2,3,4 
  fdata[3] = 2.2;
  fdata[4] = 3.3;
  memcpy( fdata + 5 , simpleVertices3D, 12 * sizeof(double) );
  printData( fdata, 3 );
  
 

  cout << "\n\nConstant 1 Example (volume of simplex)\n";
  hcubature(1, constant3D, &fdata, 3, xmin, xmax, 0, tol, tol, ERROR_INDIVIDUAL, &val, &err);
  cout << "Computed integral = " << val << " +/- " << err << "\n";
  cout << "Analytic solution = " <<       1./6             << "\n";
  cout << "Difference        = " << val - 1./6             << "\n";
  cout << "Tolerance         = " << tol                   << "\n";

    
}


void testSimple3D() {
  
  double fdata[1 + 1 + 3 + 12];
  double xmin[3] = {0,0,0}, xmax[3] = {1,1,1}, val, err, tol = 1e-5;
  
  const double vertices3D[12] = {
    0, 0, 0, // a
    3, 0, 0, // b
    0, 2, 0, // c
    0, 0, 6  // d
  };
    
  fdata[0] = 1.23;
  fdata[1] = detA( vertices3D, 3 );
  fdata[2] = 1.1; // Ignoring y for now, in entries 2,3,4
  fdata[3] = 2.2;
  fdata[4] = 3.3;
  memcpy( fdata + 1 + 1 + 3 , vertices3D, 12 * sizeof(double) );
   
  cout << "\n\nExample 2 from: http://tutorial.math.lamar.edu/Classes/CalcIII/TripleIntegrals.aspx \n";
  hcubature(1, simple3D, &fdata, 3, xmin, xmax, 0, tol, tol, ERROR_INDIVIDUAL, &val, &err);
  cout << "Computed integral = " << val << " +/- " << err << "\n";
  cout << "Analytic solution = " << 9                     << "\n";
  cout << "Difference        = " << val - 9               << "\n";
  cout << "Tolerance         = " << tol                   << "\n";
  
}

void testBetaCube() {
  
  double fdata[1 + 3] = { 1.23, 0.5, 0.5, 0 };
  double xmin[3] = {0,0,0}, xmax[3] = {1,1,1};
  double val[4];
  double err[4];
  double tol = 1e-5;
  
  hcubature(4,
	    betaCube, 
	    &fdata,
	    3,
	    xmin,
	    xmax,
	    0,
	    tol,
	    tol,
	    ERROR_INDIVIDUAL,
	    val,
	    err);
  for (int i = 0 ; i < 4 ; i ++ )  
    cout << "Computed integral = " << val[i] << " +/- " << err[i] << "\n";
  cout << "Analytic solution = " << 9                     << "\n";
  cout << "Difference        = " << val - 9               << "\n";
  cout << "Tolerance         = " << tol                   << "\n";
  
}



int main(int argc, char **argv)
{
  
  testSimple2D();
  testSingular2D();
  

  testSimpleA3D();
  testHardA3D();
  testSimplexVolume3D();
  testSimple3D();
  testBetaCube();
  
  
  return 0;
}





