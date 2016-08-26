#include <boost/math/special_functions/bessel.hpp>
#include <math.h>
using boost::math::cyl_bessel_k;

namespace dolfin {
  
  class Fundamental : public Expression
  {
  public:
    Fundamental() : Expression(), y(2), kappa(0) { }
    
    void eval(Array<double>& values, const Array<double>& x) const
    {
      /*
	I added 1E-9 to avoid ra = 0. This addition enforces *soft* thresholding
	which is know to be better than hard thresholding 
      */
      double ra = sqrt(  (x[0]-y[0])*(x[0]-y[0])  +  (x[1]-y[1])*(x[1]-y[1])  ) + 1e-12;
      values[0] = cyl_bessel_k( 0.0, kappa*ra ) / (2. * 3.14159265359);
    }
  public:
    const Array<double> y;
    double kappa;
  };
}
