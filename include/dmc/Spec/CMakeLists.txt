set(LLVM_TARGET_DEFINITIONS ParameterList.td)
mlir_tablegen(ParameterList.h.inc -gen-op-interface-decls)
mlir_tablegen(ParameterList.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(DMCParameterListIncGen)
