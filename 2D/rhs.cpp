/* 
   Please note that this code is missing a 1/2*pi factor. It is omitted since
   this factor cancels out in the computaition I'm doing. If you need
   it - just add it on your own.
*/
#include <boost/math/special_functions/bessel.hpp>
#include <math.h>
using boost::math::cyl_bessel_k;

namespace dolfin {
  
  
  class RHS : public Expression
  {
  public:
    RHS() : Expression(4), x(2), kappa(0), nu(0), factor(0) { }
    
    void eval(Array<double>& values, const Array<double>& y) const
    {
      /*
	I add 1E-13 to avoid ra = 0. This addition enforces *soft* thresholding
	which is sometimes better than hard thresholding 
      */
      double ra    = sqrt(  (x[0]-y[0])*(x[0]-y[0])  +  (x[1]-y[1])*(x[1]-y[1])  ) + 1E-13;
      double pre   = factor * pow( kappa*ra, nu );
      
      double phi   =          pre * cyl_bessel_k( nu,   kappa*ra );
      double grad  = kappa  * pre * cyl_bessel_k( nu-1, kappa*ra );
      
      double grad1 = grad   * (x[0] - y[0]) / ra;
      double grad2 = grad   * (x[1] - y[1]) / ra;
      
      values[0]    = phi    * grad1;
      values[1]    = phi    * grad2;
      values[2]    = grad1;
      values[3]    = grad2;
    }
  public:
    const Array<double> x;
    double kappa;
    double nu;
    double factor;
  };
}
