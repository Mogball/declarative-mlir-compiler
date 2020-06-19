#include "Context.h"
#include "Location.h"
#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Dynamic/DynamicType.h"
#include "dmc/Dynamic/Alias.h"
#include "dmc/Python/Polymorphic.h"

using namespace pybind11;
using namespace mlir;

namespace dmc {
namespace py {

static Type buildDynamicType(
    std::string dialectName, std::string typeName,
    const std::vector<Attribute> &params, Location loc) {
  auto *dialect = mlir::py::getMLIRContext()->getRegisteredDialect(dialectName);
  if (!dialect)
    throw std::invalid_argument{"Unknown dialect name: " + dialectName};
  auto *dynDialect = dynamic_cast<DynamicDialect *>(dialect);
  if (!dynDialect)
    throw std::invalid_argument{"Not a dynamic dialect: " + dialectName};
  auto *impl = dynDialect->lookupType(typeName);
  if (!impl)
    throw std::invalid_argument{"Unknown type '" + typeName + "' in dialect '" +
                                dialectName + "'"};
  return DynamicType::getChecked(loc, impl, params);
}

static Type getAliasedType(std::string dialectName, std::string aliasName) {
  auto *dialect = mlir::py::getMLIRContext()->getRegisteredDialect(dialectName);
  if (!dialect)
    throw std::invalid_argument{"Unknown dialect name: " + dialectName};
  auto *dynDialect = dynamic_cast<DynamicDialect *>(dialect);
  if (!dynDialect)
    throw std::invalid_argument{"Not a dynamic dialect: " + dialectName};
  auto *alias = dynDialect->lookupTypeAlias(aliasName);
  if (!alias)
    throw std::invalid_argument{"Unknown type '" + aliasName + "' in dialect '" +
                                dialectName + "'"};
  return alias->getAliasedType();
}

void exposeDynamicTypes(module &m) {
  m.def("build_dynamic_type", &buildDynamicType,
        "dialectName"_a, "typeName"_a, "params"_a = std::vector<Attribute>{},
        "location"_a = mlir::py::getUnknownLoc());

  m.def("get_aliased_type", &getAliasedType);
}

} // end namespace py
} // end namespace dmc
