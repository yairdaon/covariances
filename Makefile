run:
	./square.py
	./parallelogram.py
	./antarctica.py
	./cube.py
hpc:
	rm -rvf results.o*
	rm -rvf tests.o*
	sbatch tests

clean:
	rm -rvf *~ *.pyc cpp/*~
	rm -rvf cov/C/build
	rm -vf cov/C/*.so	
	rm -vf *.so
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
