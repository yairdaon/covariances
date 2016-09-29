# covariances
This package generates the data files used in our paper. Data 
square and parallelogram meshes is saved in text files in the
designated subdirectories data/square and data/parallelogram,
respectively. Data for the antarctica mesh cube mesh is saved 
as pvd and vtu files in their corresponding locations. 
clone the library and type "make run" to see how this goes.
The code is written in python 2 more specifically, python 2.7)
and requires the numpy and scipy packages, as well as the
FEniCS package, version 2016.1.0.

This package also includes parts of Steven Johnson's cubature package
http://ab-initio.mit.edu/wiki/index.php/Cubature , as well as tests.