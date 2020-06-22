#include "Location.h"
#include "Utility.h"

#include <mlir/IR/Module.h>
#include <llvm/Support/raw_os_ostream.h>

using namespace llvm;
using namespace pybind11;

namespace mlir {
namespace py {

std::optional<std::string> getModuleName(ModuleOp moduleOp) {
  if (auto name = moduleOp.getName())
    return name->str();
  return {};
}

} // end namespace py
} // end namespace mlir
