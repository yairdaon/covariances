#include <boost/math/special_functions/bessel.hpp>
#include <math.h>
using boost::math::cyl_bessel_k;

namespace dolfin {
  
  class Fundamental : public Expression
  {
  public:
    Fundamental() : Expression(), y(2), kappa(0), factor(0), sig2(0) { }
    
    void eval(Array<double>& values, const Array<double>& x) const
    {
      double ra = sqrt(  (x[0]-y[0])*(x[0]-y[0])  +  (x[1]-y[1])*(x[1]-y[1]) );
      if (ra == 0.0 )
	values[0] = sig2;
      else
	values[0] = factor * kappa*ra * cyl_bessel_k( 1.0, kappa*ra );
    }
  public:
    const Array<double> y;
    double kappa;
    double factor;
    double sig2;
  };
}
