#include "Location.h"
#include "Support.h"

#include <mlir/IR/Module.h>
#include <llvm/Support/raw_os_ostream.h>
#include <boost/python.hpp>

using namespace llvm;
using namespace boost::python;

namespace mlir {

std::ostream &operator<<(std::ostream &os, ModuleOp moduleOp) {
  return printToOs(os, moduleOp);
}

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

object getName(ModuleOp moduleOp) {
  if (!moduleOp) throw std::runtime_error{"ModuleOp is null"};
  if (auto name = moduleOp.getName())
    return str(name->str());
  return {};
}

} // end namespace py
} // end namespace mlir
