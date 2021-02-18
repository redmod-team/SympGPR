sources = vode/dvode_f90_m.f90 henon_mod.f90 henon.f90 

all: henon.*.so

henon.*.so: henon.pyf
	python3 -m numpy.f2py -m henon -c henon.pyf $(sources)
	
henon.pyf: henon_mod.f90 henon.f90
	python3 -m numpy.f2py henon_mod.f90 henon.f90 -m henon -h henon.pyf --overwrite-signature