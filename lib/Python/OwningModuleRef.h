#pragma once

#include <mlir/IR/Module.h>

namespace mlir {
namespace py {

std::string printModuleRef(OwningModuleRef &moduleRef);
ModuleOp getOwnedModule(OwningModuleRef &moduleRef);

} // end namespace py
} // end namespace mlir
