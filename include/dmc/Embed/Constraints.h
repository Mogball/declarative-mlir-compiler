#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Types.h>

namespace mlir {
namespace py {

void init(MLIRContext *ctx);

LogicalResult evalConstraintExpr(StringRef expr, Attribute attr);
LogicalResult evalConstraintExpr(StringRef expr, Type type);

} // end namespace py
} // end namespace mlir
