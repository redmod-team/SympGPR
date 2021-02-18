all: fieldlines.*.so

fieldlines.*.so: fieldlines.pyf fieldlines.o minpack.o
	python3 -m numpy.f2py -m fieldlines -c fieldlines.pyf fieldlines.o minpack.o

fieldlines.pyf: fieldlines.f90
	python3 -m numpy.f2py fieldlines.f90 -m fieldlines -h fieldlines.pyf --overwrite-signature
	
fieldlines.o: fieldlines.f90
	gfortran -fPIC -fbacktrace -c -g fieldlines.f90

minpack.o: minpack.f90
	gfortran -fPIC -c -march=native -O3 minpack.f90
