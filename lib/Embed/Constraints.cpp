#include "Scope.h"
#include "dmc/Embed/Constraints.h"

#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/StandardTypes.h>
#include <pybind11/embed.h>
#include <pybind11/cast.h>

using namespace pybind11;

namespace mlir {
namespace py {

/// Evaluate a single expression Python constraint on a type or attribute.
template <typename ArgT>
static LogicalResult evalConstraint(StringRef expr, const char *name,
                                    ArgT arg) {
  /// Substitute placeholder `self -> name`.
  dict fmtArgs{"self"_a = name};
  auto pyExpr = pybind11::cast(expr.str()).cast<str>().format(**fmtArgs);
  /// Wrap in a function.
  auto scope = getMainScope();
  dict funcExpr{"name"_a = name, "expr"_a = pyExpr};
  auto funcStr = "def constraint({name}): return {expr}"_s.format(**funcExpr);
  exec(funcStr, scope);
  /// Call the function.
  return success(scope["constraint"](arg).template cast<bool>());
}

LogicalResult evalConstraintExpr(StringRef expr, Attribute attr) {
  return evalConstraint(expr, "attr", attr);
}

LogicalResult evalConstraintExpr(StringRef expr, Type type) {
  return evalConstraint(expr, "type", type);
}

/// Verify that a single expression constraint has valid syntax, can be called,
/// and returns a boolean value.
template <typename ArgT>
static LogicalResult verifyConstraint(Location loc, StringRef expr,
                                      const char *name, ArgT dummy) {
  try {
    evalConstraint(expr, name, dummy);
  } catch (std::runtime_error &e) {
    return emitError(loc) << "Failed to verify constraint: " << e.what();
  }
  return success();
}

LogicalResult verifyAttrConstraint(Location loc, StringRef expr) {
  return verifyConstraint(loc, expr, "attr", UnitAttr::get(loc.getContext()));
}

LogicalResult verifyTypeConstraint(Location loc, StringRef expr) {
  return verifyConstraint(loc, expr, "type", NoneType::get(loc.getContext()));
}

} // end namespace py
} // end namespace mlir
