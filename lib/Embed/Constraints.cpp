#include "Scope.h"
#include "dmc/Embed/Constraints.h"

/// The polymorphic_type_hook must be visible so that Type and Attribute can be
/// downcasted to their appropriate derived classes.
#include "dmc/Python/Polymorphic.h"

#include <mlir/IR/Diagnostics.h>
#include <pybind11/embed.h>
#include <pybind11/cast.h>

using namespace pybind11;

namespace mlir {
namespace py {

namespace {
class ConstraintRegistry {
public:
  static ConstraintRegistry &get() {
    static ConstraintRegistry instance;
    return instance;
  }

  /// Function registers a constraint and returns the name. Throws on error.
  std::string registerConstraint(std::string expr) {
    // Substitute `{self}`
    dict fmtArgs{"self"_a = "arg"};
    auto pyExpr = pybind11::cast(expr).cast<str>().format(**fmtArgs);
    // Wrap in a function and register it in the main scope
    auto scope = getMainScope();
    std::string funcName{"anonymous_constraint_"};
    funcName += std::to_string(idx++);
    dict funcExpr{"func_name"_a = funcName, "expr"_a = pyExpr};
    auto funcStr = "def {func_name}(arg): return {expr}"_s
        .format(**funcExpr);
    exec(funcStr, scope);
    return funcName;
  }

  template <typename ArgT>
  LogicalResult evalConstraint(const std::string &funcName, ArgT arg) {
    return success(getMainScope()[funcName.c_str()](arg).template cast<bool>());
  }

private:
  ConstraintRegistry() = default;

  std::size_t idx{};
};
} // end anonymous namespace

LogicalResult registerConstraint(Location loc, StringRef expr,
                                 std::string &funcName) {
  try {
    funcName = ConstraintRegistry::get().registerConstraint(expr.str());
  } catch (const std::runtime_error &e) {
    return emitError(loc) << "Failed to create Python constraint: " << e.what();
  }
  return success();
}

LogicalResult evalConstraint(const std::string &funcName, Type type) {
  return ConstraintRegistry::get().evalConstraint(funcName, type);
}

LogicalResult evalConstraint(const std::string &funcName, Attribute attr) {
  return ConstraintRegistry::get().evalConstraint(funcName, attr);
}

} // end namespace py
} // end namespace mlir
