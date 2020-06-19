#include "Scope.h"
#include "dmc/Embed/InMemoryDef.h"
#include "dmc/Dynamic/DynamicType.h"
#include "dmc/Dynamic/DynamicDialect.h"

#include <llvm/ADT/STLExtras.h>
#include <pybind11/pybind11.h>

using namespace pybind11;

namespace dmc {
namespace py {
namespace {

std::function<void(PythonGenStream::Line &)>
paramArgs(NamedParameterRange params) {
  return [params](PythonGenStream::Line &line) {
    llvm::interleaveComma(params, line, [&](auto param) {
      line << param.getName();
    });
  };
}

// TODO wish there was a Python API I could directly call, but one is not
// provided by pybind11, so I resort to codegen.
void exposeDynamicType(module &m, DynamicTypeImpl *impl) {
  InMemoryClass cls{impl->getName(), "mlir.DynamicType", m};
  auto &s = cls.stream();
  s.line() << "def __init__(self, " << paramArgs(impl->getParamSpec())
      << "):" << incr; {
    s.line() << "super().__init__(mlir.build_dynamic_type(\""
        << impl->getDialect()->getNamespace() << "\", \"" << impl->getName()
        << "\", [" << paramArgs(impl->getParamSpec()) << "]))";
  } s.enddef();
  for (auto param : impl->getParamSpec()) {
    s.line() << "def " << param.getName() << "(self):" << incr; {
      s.line() << "return super().getParam(\"" << param.getName() << "\")";
    } s.enddef();
  }
}

} // end anonymous namespace

module exposeDialect(DynamicDialect *dialect) {
  auto m = reinterpret_borrow<module>(
      PyImport_AddModule(dialect->getNamespace().str().c_str()));
  exec("import mlir", m.attr("__dict__"));
  for (auto *ty : dialect->getTypes()) {
    exposeDynamicType(m, ty);
  }
  return m;
}

} // end namespace py
} // end namespace dmc
