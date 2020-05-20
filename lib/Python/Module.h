#include <mlir/IR/Module.h>

#include <boost/python/object.hpp>

namespace mlir {

std::ostream &operator<<(std::ostream &os, ModuleOp moduleOp);

namespace py {

/// Factory methods.
ModuleOp getModuleOp();
ModuleOp getModuleOp(Location loc);
ModuleOp getModuleOp(std::string name);
ModuleOp getModuleOp(Location loc, std::string name);

/// Getters.
boost::python::object getName(ModuleOp moduleOp);

} // end namespace py
} // end namespace mlir
