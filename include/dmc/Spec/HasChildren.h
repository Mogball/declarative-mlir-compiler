#pragma once

#include <mlir/IR/OpDefinition.h>

namespace dmc {
namespace OpTrait {

/// Check that an operation is one of the specified types.
namespace detail {
template <typename...> struct IsOneOfImpl;

template <typename OpType, typename... OpTypes>
struct IsOneOfImpl<OpType, OpTypes...> {
  bool operator()(mlir::Operation *op) {
    return llvm::isa<OpType>(op) || IsOneOfImpl<OpTypes...>{}(op);
  }
};

template <> struct IsOneOfImpl<> {
  bool operator()(mlir::Operation *) { return false; }
};
} // end namespace detail

template <typename... OpTypes>
bool isOneOf(mlir::Operation *op) {
  return detail::IsOneOfImpl<OpTypes...>{}(op);
}

/// Print a list of operation names.
namespace detail {
template <typename...> struct PrintOpNamesImpl;

template <typename OpType, typename... OpTypes>
struct PrintOpNamesImpl<OpType, OpTypes...> {
  void operator()(mlir::InFlightDiagnostic &diag) {
    diag << OpType::getOperationName() << ", ";
    PrintOpNamesImpl<OpTypes...>{}(diag);
  }
};

template <typename OpType>
struct PrintOpNamesImpl<OpType> {
  void operator()(mlir::InFlightDiagnostic &diag) {
    diag << OpType::getOperationName();
  }
};
} // end namespace detail

template <typename... OpTypes>
void printOpNames(mlir::InFlightDiagnostic &diag) {
  return detail::PrintOpNamesImpl<OpTypes...>{}(diag);
}

/// Assert that if an operation has children, the children must each be one of
/// the specified operations.
template <typename... ChildrenOpTypes>
struct HasOnlyChildren {
  /// ConcreteType must be an iteratable op.
  template <typename ConcreteType>
  struct Impl : public mlir::OpTrait::TraitBase<ConcreteType, Impl> {
  public:
    static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
      for (auto &child : llvm::cast<ConcreteType>(op)) {
        if (!isOneOf<ChildrenOpTypes...>(&child)) {
          op->emitOpError("has invalid child operation '")
              << child.getName() << "'\n";
          auto diag = child.emitOpError("must be one of [ ");
          printOpNames<ChildrenOpTypes...>(diag);
          return diag << " ]";
        }
      }
      return mlir::success();
    }
  };
};

} // end namespace OpTrait
} // end namespace dmc
