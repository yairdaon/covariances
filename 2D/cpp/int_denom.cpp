#include <boost/math/special_functions/bessel.hpp>
#include <math.h>
using boost::math::cyl_bessel_k;

namespace dolfin {
  
  class Mat : public Expression
  {
  public:
    Mat() : Expression(), x(2), nu(0), kappa(0), factor(0) { }
    
    void eval(Array<double>& values, const Array<double>& y) const
    {
      /*
	I added 1E-13 to avoid ra = 0. This addition enforces *soft* thresholding
	which is know to be better than hard thresholding 
      */
      double ra  = sqrt(  (x[0]-y[0])*(x[0]-y[0])  +  (x[1]-y[1])*(x[1]-y[1])  ) + 1E-13;
      values[0] = factor * pow( kappa*ra, nu ) * cyl_bessel_k( nu, kappa*ra );
    }
  public:
    const Array<double> x;
    double kappa;
    double nu;
    double factor;
  };
}
