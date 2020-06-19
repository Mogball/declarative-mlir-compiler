#pragma once

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Location.h>

namespace dmc {
namespace py {

mlir::LogicalResult registerConstraint(mlir::Location loc, llvm::StringRef expr,
                                       std::string &funcName);

mlir::LogicalResult evalConstraint(const std::string &funcName,
                                   mlir::Type type);
mlir::LogicalResult evalConstraint(const std::string &funcName,
                                   mlir::Attribute attr);

} // end namespace py
} // end namespace dmc
