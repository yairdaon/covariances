run:
	./square.py
	./parallelogram.py
	./antarctica.py
	./cube.py
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
