#include <boost/math/special_functions/bessel.hpp>
#include <math.h>
using boost::math::cyl_bessel_k;

namespace dolfin {
  
  class RHS : public Expression
  {
  public:
    RHS() : Expression(), x(2), nu(0), kappa(0), factor(0) { }
    
    void eval(Array<double>& values, const Array<double>& y) const
    {
      /*
	I add 1E-13 to avoid ra = 0. This addition enforces *soft* thresholding
	which is sometimes better than hard thresholding 
      */
      double ra  = sqrt(  (x[0]-y[0])*(x[0]-y[0])  +  (x[1]-y[1])*(x[1]-y[1])  ) + 1E-13;
            
      double phi = factor * pow( kappa*ra, nu ) * cyl_bessel_k( nu, kappa*ra );
      values[0]  = phi * kappa * factor * pow( kappa*ra, nu ) * cyl_bessel_k( nu-1, kappa*ra ) * (x[0] - y[0]) / ra;
    }
  public:
    const Array<double> x;
    double kappa;
    double nu;
    double factor;
  };
}
