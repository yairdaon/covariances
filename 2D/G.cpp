/* 
   Please note that this code is missing a 1/2*pi factor. It is omitted since
   this factor cancels out in the computaition I'm doing. If you need
   it - just add it on your own.
*/
#include <boost/math/special_functions/bessel.hpp>
#include <math.h>
using boost::math::cyl_bessel_k;

namespace dolfin {
  
  
  class G : public Expression
  {
  public:
    G() : Expression(2), x(2), kappa(0) { }
    
    void eval(Array<double>& values, const Array<double>& y) const
    {
      if ( abs(x[0]-y[0])<1E-14 && abs(x[1]-y[1])<1E-14 ) 
	{	
	  values[0] = .0;
	  values[1] = values[0];
	}
      else
	{
	  double ra  = sqrt(  (x[0]-y[0])*(x[0]-y[0])  +  (x[1]-y[1])*(x[1]-y[1])  );
	  values[0] = cyl_bessel_k(0, kappa*ra);
	  values[1] = values[0];
	}
    }
  public:
    const Array<double> x;
    double kappa;
  };
}
