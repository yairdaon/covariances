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

clean:
	rm -rvf *~ *.pyc cpp/*~ *.o *.so
	rm -rvf *.pdf *.log *.aux *.fls *.out *fdb_latexmk
	rm -rvf cov/C/build
	rm -vf cov/C/*.so	
	rm -rvf build
	rm -vf cov*.so

comp: mytests
	./mytests

%.o: %.cpp $(DEPS)
	$(COMPILER) -c -o $@ $< $(CFLAGS)

mytests:$(OBJ)
	$(COMPILER) -o $@ $^ $(CFLAGS)


hcubature.so:
	g++ -c -Wall -Werror -fpic hcubature.c 
	g++ -shared -o libhcubature.so hcubature.o
