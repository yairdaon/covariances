void printArray( int lines, int columns, double * a );

void printArray( int lines, int columns, const double * a );

void unpack ( void * fdata, double * kappa, double * det, double ** y, double ** vertices, int ndim );

void unpack ( void * fdata, double * kappa, double ** y );

void printData( double kappa, double det, double * y, double * vertices, int ndim );

void printData( double * fdata, int ndim );

double detA( const double * vertices, int n );


void A2D( double *v, const double * vertices );

void swap2D( const double *x, double * y );

int simple2D( unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval); 

int singular2D( unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval);
  
int beta2D( unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval);


void A3D( double *v, const double * vertices );

double dist3D( double * x, double * y );

int closest( const double * x, const double * p );

void swap3D( const double *x, double * y );

int beta3D   ( unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval);

int simple3D( unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval);

int constant3D( unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval);

int betaCube( unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval);
  
