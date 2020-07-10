#pragma once

#include "SpecKinds.h"
#include "SpecAttrBase.h"

#include <mlir/IR/Block.h>

/// Forward declarations.
namespace mlir {
class OpAsmParser;
class OpAsmPrinter;
};

namespace dmc {
class OpSuccessor;

namespace SpecSuccessor {
bool is(mlir::Attribute base);
mlir::LogicalResult delegateVerify(mlir::Attribute base, mlir::Block *block);
/// TODO Instead of avoiding Dialect::printAttribute, use it.
std::string toString(mlir::Attribute opSucc);
} // end namespace SpecSuccessor

/// Match any successor.
class AnySuccessor : public mlir::Attribute::AttrBase<
    AnySuccessor, mlir::Attribute, mlir::AttributeStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getName() { return "Any"; }
  static bool kindof(unsigned kind) { return kind == SpecSuccessor::Any; }

  static AnySuccessor get(mlir::MLIRContext *ctx) {
    return Base::get(ctx, SpecSuccessor::Any);
  }

  inline mlir::LogicalResult verify(mlir::Block *) { return mlir::success(); }

  static Attribute parse(mlir::OpAsmParser &parser);
  void print(llvm::raw_ostream &os);
};

/// Variadic successors.
class VariadicSuccessor : public mlir::Attribute::AttrBase<
    VariadicSuccessor, mlir::Attribute, detail::OneAttrStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getName() { return "Variadic"; }
  static bool kindof(unsigned kind) { return kind == SpecSuccessor::Variadic; }

  static VariadicSuccessor getChecked(mlir::Location loc,
                                      mlir::Attribute succConstraint);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, mlir::Attribute succConstraint);

  mlir::LogicalResult verify(mlir::Block *block);

  static Attribute parse(mlir::OpAsmParser &parser);
  void print(llvm::raw_ostream &os);
};

namespace impl {
mlir::LogicalResult verifySuccessorConstraints(mlir::Operation *op,
                                               OpSuccessor opSuccs);
} // end namespace impl

} // end namespace dmc
