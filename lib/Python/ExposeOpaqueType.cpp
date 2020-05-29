#include "Location.h"
#include "Utility.h"
#include "Context.h"
#include "Identifier.h"
#include "Type.h"
#include "Expose.h"

using namespace pybind11;

namespace mlir {
namespace py {

template <typename FcnT> auto nullcheck(FcnT fcn) {
  return ::nullcheck(fcn, "opaque type");
}

OpaqueType getOpaqueType(const std::string &dialect,
                         const std::string &typeData) {
  auto id = getIdentifierChecked(dialect);
  if (failed(OpaqueType::verifyConstructionInvariants(
        getUnknownLoc(), id, typeData)))
    throw std::invalid_argument{"Bad opaque type arguments"};
  return OpaqueType::get(id, typeData, getMLIRContext());
}

void exposeOpaqueType(pybind11::module &m, TypeClass &type) {
  class_<OpaqueType>(m, "OpaqueType", type)
      .def(init(&getOpaqueType))
      .def_property_readonly("dialectNamespace", nullcheck([](OpaqueType ty) {
        return ty.getDialectNamespace().str();
      }))
      .def_property_readonly("typeData", nullcheck([](OpaqueType ty) {
        return ty.getTypeData().str();
      }));
}

} // end namespace py
} // end namespace mlir
