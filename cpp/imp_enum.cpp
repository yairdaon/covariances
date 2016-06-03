#include <boost/math/special_functions/bessel.hpp>
#include <math.h>
using boost::math::cyl_bessel_k;

namespace dolfin {
  
  class Enumerator : public Expression
  {
  public:
    Enumerator() : Expression(2), x(2), kappa(0), factor(0) { }
    
    void eval(Array<double>& values, const Array<double>& y) const
    {
      /*
	I add 1E-13 to avoid ra = 0. This addition enforces *soft* thresholding
	which is sometimes better than hard thresholding 
      */
      double ra    = sqrt(  (x[0]-y[0])*(x[0]-y[0])  +  (x[1]-y[1])*(x[1]-y[1])  ) + 1E-13;
      
      double tmp = kappa * cyl_bessel_k( 0.0, kappa*ra ) * cyl_bessel_k( 1.0, kappa*ra ) / ra;
      values[0]  = tmp * (x[0] - y[0]);
      values[1]  = tmp * (x[1] - y[1]);
      
    }
  public:
    const Array<double> x;
    double kappa;
    double factor;
  };
}
