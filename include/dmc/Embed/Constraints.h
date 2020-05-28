#pragma once

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Location.h>

namespace mlir {
namespace py {

LogicalResult evalConstraintExpr(StringRef expr, Attribute attr);
LogicalResult evalConstraintExpr(StringRef expr, Type type);

LogicalResult verifyAttrConstraint(Location loc, StringRef expr);
LogicalResult verifyTypeConstraint(Location loc, StringRef expr);

} // end namespace py
} // end namespace mlir
