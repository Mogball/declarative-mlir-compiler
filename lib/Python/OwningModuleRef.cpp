#include "Utility.h"

#include <mlir/IR/Module.h>

namespace mlir {
namespace py {

std::string printModuleRef(OwningModuleRef &moduleRef) {
  if (!moduleRef)
    throw std::invalid_argument{"module is null"};
  return StringPrinter<ModuleOp>{}(*moduleRef);
}

ModuleOp getOwnedModule(OwningModuleRef &moduleRef) {
  return *moduleRef;
}

} // end namespace py
} // end namespace mlir
