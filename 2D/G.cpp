/* 
   Please note that this code is missing a 1/2*pi factor. It is omitted since
   this factor cancels out in the computaition I'm doing. If you need
   it - just add it on your own.
*/
#include <boost/math/special_functions/bessel.hpp>
#include <math.h>
using boost::math::cyl_bessel_k;

namespace dolfin {
  
  
  class MyCppExpression : public Expression
  {
  public:
    MyCppExpression() : Expression(), x(2), kappa() { }
    
    void eval(Array<double>& values, const Array<double>& y) const
    {
      if ( abs(x[0]-y[0])<1E-12 && abs(x[1]-y[1])<1E-12 ) 
	{	
	  values[0] = 0.0;
	}
      else
	{
	  double r  = sqrt(  (x[0]-y[0])*(x[0]-y[0])  +  (x[1]-y[1])*(x[1]-y[1])  );
	  values[0] = cyl_bessel_k(0, kappa*r);
	}
    }
  public:
    const Array<double> x;
    double kappa;
  };
}
