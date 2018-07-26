all:
	wget https://download.open-mpi.org/release/hwloc/v2.0/hwloc-2.0.1.tar.bz2
	tar -xf hwloc-2.0.1.tar.bz2
	rm hwloc-2.0.1.tar.bz2
	cd hwloc-2.0.1 && \
	./configure --prefix=${PWD} && \
	make -j && make install
	rm -rf hwloc-2.0.1
	g++ -fopenmp -Iinclude -o test_omp test_omp.cpp -Llib -lhwloc

