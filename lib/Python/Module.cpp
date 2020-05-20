#include <mlir/IR/Module.h>
#include <llvm/Support/raw_os_ostream.h>

using namespace llvm;

namespace mlir {

std::ostream &operator<<(std::ostream &os, ModuleOp moduleOp) {
  raw_os_ostream rawOs{os};
  moduleOp.print(rawOs);
  return os;
}

} // end namespace mlir
