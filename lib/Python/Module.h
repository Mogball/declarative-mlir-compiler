#include <mlir/IR/Module.h>

#include <pybind11/stl.h>

namespace mlir {
namespace py {

/// Factory methods.
ModuleOp getModuleOp();
ModuleOp getModuleOp(Location loc);
ModuleOp getModuleOp(std::string name);
ModuleOp getModuleOp(Location loc, std::string name);

/// Getters.
std::optional<std::string> getName(ModuleOp moduleOp);

} // end namespace py
} // end namespace mlir
