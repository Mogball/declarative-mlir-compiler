#pragma once

#include <mlir/IR/Module.h>

namespace mlir {

std::ostream &operator<<(std::ostream &os, const OwningModuleRef &moduleRef);

namespace py {

ModuleOp getOwnedModule(OwningModuleRef &moduleRef);

} // end namespace py
} // end namespace mlir
