CFLAGS=-Ofast -g -flto
FILE=fannkuch.lua

main: main.o impl.o builtins.o
	clang++ main.o impl.o builtins.o -o main $(CFLAGS)

builtins.o: builtins.cpp lib.h
	clang++ -c -std=c++17 builtins.cpp -o builtins.o $(CFLAGS)

impl.o: impl.cpp lib.h
	clang++ -c -std=c++17 impl.cpp -o impl.o $(CFLAGS)

main.s: mainopt.ll
	clang -S mainopt.ll $(CFLAGS) -o main.s

main.o: mainopt.ll
	clang -c mainopt.ll -o main.o $(CFLAGS)

mainopt.ll: main.ll
	clang -S -emit-llvm main.ll -o mainopt.ll $(CFLAGS)

main.ll: main.mlir
	mlir-translate -mlir-to-llvmir main.mlir -o main.ll

main.mlir: luac.py $(FILE) lua.mlir lib.mlir
	python3 luac.py $(FILE) > main.mlir

clean:
	rm -f *.o
	rm -f *.ll
	rm -f main.mlir
