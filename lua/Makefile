CFLAGS=-g -O2

main: main.o impl.o start.o builtins.o
	clang++ main.o impl.o start.o builtins.o -o main $(CFLAGS)

builtins.o: builtins.cpp
	clang++ -c -std=c++17 builtins.cpp -o builtins.o $(CFLAGS)

start.o: main.c
	clang -c main.c -o start.o $(CFLAGS)

impl.o: impl.cpp
	clang++ -c -std=c++17 impl.cpp -o impl.o $(CFLAGS)

main.o: mainopt.ll
	clang -c mainopt.ll -o main.o $(CFLAGS)

mainopt.ll: main.ll
	clang -S -emit-llvm main.ll -o mainopt.ll $(CFLAGS)

main.ll: main.mlir
	mlir-translate -mlir-to-llvmir main.mlir -o main.ll

main.mlir: luac.py perf.lua
	python3 luac.py perf.lua > main.mlir
