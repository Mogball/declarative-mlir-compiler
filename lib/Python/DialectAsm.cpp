#include "dmc/Python/DialectAsm.h"
#include "dmc/Dynamic/DynamicType.h"
#include "dmc/Dynamic/DynamicAttribute.h"

#include <pybind11/pybind11.h>

using namespace pybind11;
using namespace mlir;

namespace dmc {
namespace py {

TypeWrap::TypeWrap(DynamicType type)
    : params{type.getParams()},
      paramSpec{type.getDynImpl()->getParamSpec()} {}

TypeWrap::TypeWrap(DynamicAttribute attr)
    : params{attr.getParams()},
      paramSpec{attr.getDynImpl()->getParamSpec()} {}

void exposeTypeWrap(module &m) {
  class_<TypeWrap>(m, "TypeWrap")
      .def("getParameter", [](TypeWrap &wrap, std::string name) {
        for (auto [param, spec] : llvm::zip(wrap.getParams(), wrap.getSpec())) {
          if (spec.getName() == name)
            return param;
        }
        throw std::invalid_argument{"Unknown parameter name: " + name};
      });

  class_<TypeResultWrap>(m, "TypeResultWrap")
      .def("append", [](TypeResultWrap &wrap, Attribute attr) {
        wrap.getImpl().push_back(attr);
      });
}

} // end namespace py
} // end namespace dmc
