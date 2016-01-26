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
      values[0] = 1.0;
    }
    public:
    const Array<double> x;
    double kappa;
    int nu;
    double factor;
  };
}
