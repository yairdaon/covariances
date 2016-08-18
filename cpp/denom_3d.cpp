#include <boost/math/special_functions/bessel.hpp>
#include <math.h>
using boost::math::cyl_bessel_k;

namespace dolfin {
  
  class Denominator : public Expression
  {
  public:
    Denominator() : Expression(), y(3), kappa(0), factor(0) { }
    
    void eval(Array<double>& values, const Array<double>& x) const
    {
      /*
	I added 1E-9 to avoid ra = 0. This addition enforces *soft* thresholding
	which is know to be better than hard thresholding 
      */
      double ra = sqrt(  (x[0]-y[0])*(x[0]-y[0])  + 
			 (x[1]-y[1])*(x[1]-y[1])  +
			 (x[2]-y[2])*(x[2]-y[2])  ) + 1E-9;
      values[0] = 2.0 * cyl_bessel_k( 0.5, kappa*ra ) * exp( -kappa*ra ) / sqrt( ra );
    }

  public:
    const Array<double> y;
    double kappa;
    double factor;
  };
}
