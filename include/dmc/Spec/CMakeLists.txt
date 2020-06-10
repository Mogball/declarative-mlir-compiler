set(LLVM_TARGET_DEFINITIONS ParameterList.td)
mlir_tablegen(ParameterList.h.inc -gen-op-interface-decls)
mlir_tablegen(ParameterList.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(DMCParameterListIncGen)

set(LLVM_TARGET_DEFINITIONS ReparseOpInterface.td)
mlir_tablegen(ReparseOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(ReparseOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(DMCReparseOpInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS FormatOp.td)
mlir_tablegen(FormatOp.h.inc -gen-op-interface-decls)
mlir_tablegen(FormatOp.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(DMCFormatOpIncGen)
