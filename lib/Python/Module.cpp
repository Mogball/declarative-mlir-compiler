#include "Location.h"
#include "Utility.h"

#include <mlir/IR/Module.h>
#include <llvm/Support/raw_os_ostream.h>

using namespace llvm;
using namespace pybind11;

namespace mlir {
namespace py {

ModuleOp getModuleOp() {
  return ModuleOp::create(getUnknownLoc());
}

ModuleOp getModuleOp(Location loc) {
  return ModuleOp::create(loc);
}

ModuleOp getModuleOp(std::string name) {
  return ModuleOp::create(getUnknownLoc(), StringRef{name});
}

ModuleOp getModuleOp(Location loc, std::string name) {
  return ModuleOp::create(loc, StringRef{name});
}

std::optional<std::string> getName(ModuleOp moduleOp) {
  if (auto name = moduleOp.getName())
    return name->str();
  return {};
}

} // end namespace py
} // end namespace mlir
