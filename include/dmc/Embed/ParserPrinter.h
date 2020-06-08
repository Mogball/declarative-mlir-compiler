#pragma once

#include <string>

namespace mlir {
class OpAsmParser;
class OpAsmPrinter;
class Operation;
struct OperationState;
} // end namespace mlir

namespace dmc {
class DynamicOperation;
namespace py {
bool execParser(const std::string &name, mlir::OpAsmParser &parser,
                mlir::OperationState &result);
void execPrinter(const std::string &name, mlir::OpAsmPrinter &printer,
                 mlir::Operation *op, DynamicOperation *spec);
} // end namespace py
} // end namespace dmc
