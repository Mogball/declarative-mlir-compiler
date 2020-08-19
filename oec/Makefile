main: main.o laplace.o
	clang++ main.o laplace.o -o main -fPIE -L$(LD_LIBRARY_PATH) -lcuda-runtime-wrappers -lcuda

main.o: main.cpp
	clang++ -O3 -c main.cpp -o main.o -fPIE -I/opt/cuda/include -g

laplace.o: laplace.ll
	clang -c laplace.ll -o laplace.o -fPIE

laplace.ll: laplace.mlir.lowered
	mlir-translate --mlir-to-llvmir laplace.mlir.lowered -o laplace.ll

laplace.mlir.lowered: laplace.mlir
	oec-opt --stencil-shape-inference --convert-stencil-to-std --cse --parallel-loop-tiling='parallel-loop-tile-sizes=128,1,1' --canonicalize --test-gpu-greedy-parallel-loop-mapping --convert-parallel-loops-to-gpu --canonicalize --lower-affine --convert-scf-to-std --stencil-kernel-to-cubin laplace.mlir > laplace.mlir.lowered

clean:
	rm -f *.s
	rm -f *.o
	rm -f *.ll
	rm -f *.lowered
