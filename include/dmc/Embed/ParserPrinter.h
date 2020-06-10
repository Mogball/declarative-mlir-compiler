#pragma once

#include <string>
#include <vector>

namespace mlir {
class OpAsmParser;
class OpAsmPrinter;
class Operation;
struct OperationState;
class DialectAsmParser;
class DialectAsmPrinter;
class Attribute;
} // end namespace mlir

namespace dmc {
class DynamicOperation;
namespace py {
bool execParser(const std::string &name, mlir::OpAsmParser &parser,
                mlir::OperationState &result);
void execPrinter(const std::string &name, mlir::OpAsmPrinter &printer,
                 mlir::Operation *op, DynamicOperation *spec);
bool execParser(const std::string &name, mlir::DialectAsmParser &parser,
                std::vector<mlir::Attribute> &result);
template <typename DynamicT>
void execPrinter(const std::string &name, mlir::DialectAsmPrinter &printer,
                 DynamicT type);
} // end namespace py
} // end namespace dmc
