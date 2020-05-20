#include "Module.h"

using namespace llvm;

namespace mlir {

/// Print a ModuleOp held by an OwningModuleRef.
std::ostream &operator<<(std::ostream &os, const OwningModuleRef &moduleRef) {
  return os << *moduleRef;
}

} // end namespace mlir
