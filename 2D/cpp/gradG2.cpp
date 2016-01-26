/* 
   Please note that this code is missing a 1/2*pi factor. It is omitted since
   this factor cancels out in the computaition I'm doing. If you need
   it - just add it on your own.
*/
#include <boost/math/special_functions/bessel.hpp>
#include <math.h>
using boost::math::cyl_bessel_k;

namespace dolfin {
  
  
  class GradG : public Expression
  {
  public:
    GradG() : Expression(0),  kappa(0), x(2) { }
    
    void eval(Array<double>& values, const Array<double>& y) const
    {
      /*
	I added 1E-13 to avoid ra = 0. This addition enforces *soft* thresholding
	which is know to be better than hard thresholding 
      */
      double ra  = sqrt(  (x[0]-y[0])*(x[0]-y[0])  +  (x[1]-y[1])*(x[1]-y[1])  ) + 1E-13;
      values[0]  = -kappa * cyl_bessel_k( 0, kappa*ra ) * cyl_bessel_k( 1, kappa*ra ) * (x[1]-y[1]) / ra;
      values[1]  =  tmp ; 
    }
  public:
    double kappa;
    const Array<double> x;
  };
}
