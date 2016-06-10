#include <boost/math/special_functions/bessel.hpp>
#include <math.h>
using boost::math::cyl_bessel_k;

namespace dolfin {
  
  class Enumerator : public Expression
  {
  public:
    Enumerator() : Expression(2), y(2), kappa(0), factor(0) { }
    
    void eval(Array<double>& values, const Array<double>& x) const
    {
      /*
	I add 1E-13 to avoid ra = 0. This addition enforces *soft* thresholding
	which is sometimes better than hard thresholding 
      */
      double ra  = sqrt(  (x[0]-y[0])*(x[0]-y[0])  +  (x[1]-y[1])*(x[1]-y[1])  ) + 1E-13;
      
      double phi1 = cyl_bessel_k( 0.0, kappa*ra );
      double phi2 = cyl_bessel_k( 1.0, kappa*ra );
      double tot  = kappa * kappa * (phi1*phi1 + phi2*phi2);

      values[0] =  tot * (y[0] - x[0]);
      values[1] =  tot * (y[1] - x[1]);
    }
  public:
    const Array<double> y;
    double kappa;
    double factor;
  };
}
