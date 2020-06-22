#include "Scope.h"
#include "dmc/Dynamic/DynamicOperation.h"
#include "dmc/Dynamic/DynamicType.h"
#include "dmc/Dynamic/DynamicAttribute.h"
#include "dmc/Python/OpAsm.h"
#include "dmc/Python/DialectAsm.h"

#include <mlir/IR/Operation.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;
using namespace pybind11;

// is_copy_constructible is true for OperationState even though it contains
// vector<unique_ptr>, so explicitly mark it as non copy constructible.
template <> struct std::is_copy_constructible<OperationState>
    : public std::false_type {};

namespace dmc {
namespace py {

bool execParser(const std::string &name, OpAsmParser &parser,
                OperationState &result) {
  constexpr auto parser_policy = return_value_policy::reference;
  ensureBuiltins(getInternalModule());
  auto fcn = getInternalScope()[name.c_str()];
  return fcn.operator()<parser_policy>(parser, result).cast<bool>();
}

void execPrinter(const std::string &name, OpAsmPrinter &printer, Operation *op,
                 DynamicOperation *spec) {
  constexpr auto printer_policy = return_value_policy::reference;
  ensureBuiltins(getInternalModule());
  auto fcn = getInternalScope()[name.c_str()];
  OperationWrap wrap{op, spec};
  fcn.operator()<printer_policy>(printer, &wrap);
}

bool execParser(const std::string &name, DialectAsmParser &parser,
                std::vector<Attribute> &result) {
  constexpr auto parser_policy = return_value_policy::reference;
  ensureBuiltins(getInternalModule());
  auto fcn = getInternalScope()[name.c_str()];
  TypeResultWrap wrap{result};
  return fcn.operator()<parser_policy>(parser, wrap).cast<bool>();
}

template <typename DynamicT>
void execPrinter(const std::string &name, DialectAsmPrinter &printer,
                 DynamicT t) {
  constexpr auto printer_policy = return_value_policy::reference;
  ensureBuiltins(getInternalModule());
  auto fcn = getInternalScope()[name.c_str()];
  TypeWrap wrap{t};
  fcn.operator()<printer_policy>(printer, &wrap);
}

template void execPrinter(const std::string &name, DialectAsmPrinter &printer,
                          DynamicType type);
template void execPrinter(const std::string &name, DialectAsmPrinter &printer,
                          DynamicAttribute attr);

} // end namespace py
} // end namespace dmc
