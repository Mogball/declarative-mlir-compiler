#include "Scope.h"
#include "dmc/Embed/Constraints.h"

#include <pybind11/embed.h>
#include <pybind11/cast.h>

using namespace pybind11;

namespace mlir {
namespace py {

template <typename ArgT>
static LogicalResult evalConstraint(StringRef expr, const char *name,
                                    ArgT arg) {
  /// Substitute placeholder `$_self -> name`.
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

} // end namespace py
} // end namespace mlir
