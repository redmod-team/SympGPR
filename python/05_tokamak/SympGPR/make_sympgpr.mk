FFLAGS = -fPIC -c -march=native -O3 -finit-real=nan

all: sympgpr.*.so

sympgpr.*.so: sympgpr.pyf sympgpr.o fieldlines.o kernels.o minpack.o
	python3 -m numpy.f2py -m sympgpr -c $^

sympgpr.pyf: sympgpr.f90
	python3 -m numpy.f2py sympgpr.f90 -m sympgpr -h sympgpr.pyf --overwrite-signature

kernels.o: kernels.f90
	gfortran $(FFLAGS) $<

sympgpr.o: sympgpr.f90 fieldlines.o
	gfortran $(FFLAGS) $<

fieldlines.o: fieldlines.f90
	gfortran $(FFLAGS) $<

minpack.o: minpack.f90
	gfortran $(FFLAGS) $<
