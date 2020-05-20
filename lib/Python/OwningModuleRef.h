#pragma once

#include <mlir/IR/Module.h>

namespace mlir {

std::ostream &operator<<(std::ostream &os, const OwningModuleRef &moduleRef);

} // end namespace mlir
