FC = gfortran
FFLAGS = -Wall -march=native -O2 -g -fbacktrace
PYTHON = python3
NAME = kernels_sum

all: $(NAME).f90
	$(PYTHON) -m numpy.f2py -m $(NAME) -c $(NAME).f90 --f90flags='$(FFLAGS)' -lgomp

$(NAME).f90: init_func.py
	$(PYTHON) init_func.py

clean:
	rm -f *.x *.so *.o *.mod *.pyf *.pyd *.dylib *.dll $(NAME).f90
