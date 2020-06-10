#include "dmc/Python/DialectAsm.h"
#include "dmc/Dynamic/DynamicType.h"
#include "dmc/Dynamic/DynamicAttribute.h"

#include <pybind11/pybind11.h>

using namespace pybind11;

namespace dmc {
namespace py {

TypeWrap::TypeWrap(DynamicType type)
    : params{type.getParams()},
      paramSpec{type.getTypeImpl()->getParamSpec()} {}

TypeWrap::TypeWrap(DynamicAttribute attr)
    : params{attr.getParams()},
      paramSpec{attr.getAttrImpl()->getParamSpec()} {}

void exposeTypeWrap(module &m) {
  class_<TypeWrap>(m, "TypeWrap")
      .def("getParameter", [](TypeWrap &wrap, std::string name) {
        for (auto [param, spec] : llvm::zip(wrap.getParams(), wrap.getSpec())) {
          if (spec.getName() == name)
            return param;
        }
        throw std::invalid_argument{"Unknown parameter name: " + name};
      });
}

} // end namespace py
} // end namespace dmc
