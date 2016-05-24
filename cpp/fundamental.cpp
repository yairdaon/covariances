#include <boost/math/special_functions/bessel.hpp>
#include <math.h>
using boost::math::cyl_bessel_k;

namespace dolfin {
  
  class Fundamental : public Expression
  {
  public:
    Fundamental() : Expression(), x(2), kappa(0), factor(0) { }
    
    void eval(Array<double>& values, const Array<double>& y) const
    {
      /*
	I added 1E-13 to avoid ra = 0. This addition enforces *soft* thresholding
	which is know to be better than hard thresholding 
      */
      double ra = sqrt(  (x[0]-y[0])*(x[0]-y[0])  +  (x[1]-y[1])*(x[1]-y[1])  ) + 1E-13;
      values[0] = factor * kappa*ra * cyl_bessel_k( 1.0, kappa*ra );
    }
  public:
    const Array<double> x;
    double kappa;
    double factor;
  };
}
