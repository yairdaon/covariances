COMPILER= g++
CFLAGS= -I. -lm
DEPS = header.h cubature.h converged.h vwrapper.h aux.h
OBJ = mytests.o helper.o hcubature.o 

run:
	python simulate.py square
	python simulate.py parallelogram std
	python simulate.py parallelogram radial
	python simulate.py antarctica std
	python simulate.py antarctica radial	
	python simulate.py cube std
	python simulate.py cube radial
	python boundary.py

cube:
	python simulate.py cube std
	python simulate.py cube radial
 
tmp:
	rm -rvf tmp.o*
	sbatch tmp

tests:
	rm -rvf test.o*
	sbatch test

simple:
	rm -rvf results_simple.o*
	sbatch simple

parallelogram:
	rm -rvf results_parallelogram.o*
	sbatch parall

full: 
	rm -rvf results_full.o*
	sbatch full

clean:
	rm -rvf *~ *.pyc cpp/*~ *.o *.so
	rm -rvf cov/C/build
	rm -vf cov/C/*.so	
	rm -rvf build
	rm -vf cov*.so


build7:
	python2.7 cov/C/setup.py build_ext --inplace
	mv build cov/C
	mv *.so cov

build6:
	python2.7 cov/C/setup.py build_ext --inplace
	mv build cov/C
	mv *.so cov

comp: mytests
	./mytests

%.o: %.cpp $(DEPS)
	$(COMPILER) -c -o $@ $< $(CFLAGS)

mytests:$(OBJ)
	$(COMPILER) -o $@ $^ $(CFLAGS)


hcubature.so:
	g++ -c -Wall -Werror -fpic hcubature.c 
	g++ -shared -o libhcubature.so hcubature.o
