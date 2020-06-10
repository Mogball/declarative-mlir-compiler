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

namespace dmc {
namespace py {

bool execParser(const std::string &name, OpAsmParser &parser,
                OperationState &result) {
  constexpr auto parser_policy = return_value_policy::reference;
  auto fcn = getMainScope()[name.c_str()];
  ResultWrap resultWrap{result};
  return fcn.operator()<parser_policy>(parser, resultWrap).cast<bool>();
}

void execPrinter(const std::string &name, OpAsmPrinter &printer, Operation *op,
                 DynamicOperation *spec) {
  constexpr auto printer_policy = return_value_policy::reference;
  auto fcn = getMainScope()[name.c_str()];
  OperationWrap opWrap{op, spec};
  fcn.operator()<printer_policy>(printer, &opWrap);
}

template <typename DynamicT>
void execPrinter(const std::string &name, DialectAsmPrinter &printer,
                 DynamicT t) {
  constexpr auto printer_policy = return_value_policy::reference;
  auto fcn = getMainScope()[name.c_str()];
  TypeWrap wrap{t};
  fcn.operator()<printer_policy>(printer, &wrap);
}

template void execPrinter(const std::string &name, DialectAsmPrinter &printer,
                          DynamicType type);
template void execPrinter(const std::string &name, DialectAsmPrinter &printer,
                          DynamicAttribute attr);

} // end namespace py
} // end namespace dmc
