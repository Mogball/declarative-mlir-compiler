#include <mlir/IR/Module.h>

#include <pybind11/stl.h>

namespace mlir {
namespace py {

/// Getters.
std::optional<std::string> getModuleName(ModuleOp moduleOp);

} // end namespace py
} // end namespace mlir
