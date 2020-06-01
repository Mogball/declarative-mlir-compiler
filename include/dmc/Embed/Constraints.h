#pragma once

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Location.h>

namespace mlir {
namespace py {

LogicalResult registerConstraint(Location loc, StringRef expr,
                                 std::string &funcName);

LogicalResult evalConstraint(const std::string &funcName, Type type);
LogicalResult evalConstraint(const std::string &funcName, Attribute attr);

} // end namespace py
} // end namespace mlir
