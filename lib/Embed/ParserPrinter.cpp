#include "Scope.h"
#include "dmc/Dynamic/DynamicOperation.h"
#include "dmc/Python/OpAsm.h"

#include <mlir/IR/Operation.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;
using namespace pybind11;

namespace dmc {
namespace py {

bool execParser(const std::string &name, OpAsmParser &parser,
                OperationState &result) {
  return false;
}

void execPrinter(const std::string &name, OpAsmPrinter &printer, Operation *op,
                 DynamicOperation *spec) {
  OperationWrap opWrap{op, spec};
  getMainScope()[name.c_str()].operator()<return_value_policy::reference>(
      printer, &opWrap);
}

} // end namespace py
} // end namespace dmc
