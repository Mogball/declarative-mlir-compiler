#include "Support.h"

#include <mlir/IR/Module.h>
#include <llvm/Support/raw_os_ostream.h>

using namespace llvm;

namespace mlir {

std::ostream &operator<<(std::ostream &os, ModuleOp moduleOp) {
  return printToOs(os, moduleOp);
}

} // end namespace mlir
