FC = gfortran
FFLAGS = -Wall -march=native -O3 -g -fbacktrace
PYTHON = python3
NAME = kernels

all: $(NAME).f90
	$(PYTHON) -m numpy.f2py -m $(NAME) -c $(NAME).f90 --f90flags='$(FFLAGS)' -lgomp

$(NAME).f90: init_func.py
	$(PYTHON) init_func.py
