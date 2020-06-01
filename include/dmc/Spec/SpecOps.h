#pragma once

#include "ParameterList.h"
#include "HasChildren.h"
#include "ReparseOpInterface.h"
#include "dmc/Traits/OpTrait.h"

#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/FunctionSupport.h>

namespace dmc {

/// Forward declarations.
class DialectTerminatorOp;
class OperationOp;
class TypeOp;
class AttributeOp;
class AliasOp;

/// Top-level Op in the SpecDialect which defines a dialect:
///
/// dmc.Dialect @MyDialect {foldHook = @myFoldHook, ...} {
///   ...
/// }
///
/// Captured in the Op region are the Dialect Operations. The attributes are
/// used to configure the generated DynamicDialect.
class DialectOp
    : public mlir::Op<
          DialectOp, mlir::OpTrait::ZeroOperands, mlir::OpTrait::ZeroResult,
          mlir::OpTrait::IsIsolatedFromAbove, mlir::OpTrait::SymbolTable,
          mlir::OpTrait::SingleBlockImplicitTerminator<DialectTerminatorOp>::Impl,
          mlir::SymbolOpInterface::Trait,
          dmc::OpTrait::HasOnlyChildren<
              DialectTerminatorOp, OperationOp, TypeOp, AttributeOp,
              AliasOp>::Impl> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "dmc.Dialect"; }
  static void build(mlir::OpBuilder &builder, mlir::OperationState &result,
                    llvm::StringRef name);

  /// Operation hooks.
  static mlir::ParseResult parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result);
  void print(mlir::OpAsmPrinter &printer);
  mlir::LogicalResult verify();

  /// Getters.
  bool allowsUnknownOps();
  bool allowsUnknownTypes();

  mlir::Region &getBodyRegion();
  mlir::Block *getBody();

  /// Get children ops of the given kind.
  template <typename T> auto getOps() {
    return getBody()->getOps<T>();
  }

  /// Iteration.
  mlir::Block::iterator begin() { return getBody()->begin(); }
  mlir::Block::iterator end() { return getBody()->end(); }

private:
  /// Attributes.
  static void buildDefaultValuedAttrs(mlir::OpBuilder &builder,
                                      mlir::OperationState &result);

  static inline llvm::StringRef getAllowUnknownOpsAttrName() {
    return "allow_unknown_ops";
  }
  static inline llvm::StringRef getAllowUnknownTypesAttrName() {
    return "allow_unknown_types";
  }
};

/// Special terminator Op for DialectOp.
class DialectTerminatorOp
    : public mlir::Op<DialectTerminatorOp,
                      mlir::OpTrait::ZeroOperands, mlir::OpTrait::ZeroResult,
                      mlir::OpTrait::HasParent<DialectOp>::Impl,
                      mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName() { return "dmc.DialectTerminator"; }
  static inline void build(mlir::OpBuilder &, mlir::OperationState &) {}
};

/// Dialect Op definition Op. This Op captures information about an operation:
///
/// dmc.Op @MyOpA(!dmc.Any, !dmc.AnyOf<!dmc.I<32>, !dmc.F<32>>) ->
///     (!dmc.AnyFloat, !dmc.AnyInteger)
///     { attr0 = !dmc.Any, attr1 = !dmc.StrAttr }
///     config { parser = @MyOpAParser, printer = @MyOpAPrinter
///              traits = [@IsCommutative]}
///
class OperationOp
    : public mlir::Op<OperationOp,
                      mlir::OpTrait::ZeroOperands, mlir::OpTrait::ZeroResult,
                      mlir::OpTrait::IsIsolatedFromAbove, mlir::OpTrait::FunctionLike,
                      mlir::OpTrait::HasParent<DialectOp>::Impl,
                      mlir::SymbolOpInterface::Trait,
                      mlir::dmc::ReparseOpInterface::Trait> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "dmc.Op"; }

  static void build(mlir::OpBuilder &builder, mlir::OperationState &result,
                    llvm::StringRef name, mlir::FunctionType type,
                    llvm::ArrayRef<mlir::NamedAttribute> opAttrs,
                    llvm::ArrayRef<mlir::Attribute> opRegions,
                    llvm::ArrayRef<mlir::NamedAttribute> config);

  /// Operation hooks.
  static mlir::ParseResult parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result);
  void print(mlir::OpAsmPrinter &printer);
  mlir::LogicalResult verify();

  /// Getters.
  mlir::DictionaryAttr getOpAttrs();
  mlir::ArrayAttr getOpRegions();
  OpTraitsAttr getOpTraits();

  bool isTerminator();
  bool isCommutative();
  bool isIsolatedFromAbove();

  unsigned getNumOperands();
  unsigned getNumResults();

  /// Reparse types and attributes.
  mlir::ParseResult reparse();

private:
  /// Hooks for FunctionLike
  friend class mlir::OpTrait::FunctionLike<OperationOp>;
  unsigned getNumFuncArguments() { return getType().getInputs().size(); }
  unsigned getNumFuncResults() { return getType().getResults().size(); }
  mlir::LogicalResult verifyType();

  /// Replace the Op type.
  void setOpType(mlir::FunctionType opTy);
  /// Replace the Op attributes.
  void setOpAttrs(mlir::DictionaryAttr opAttrs);
  /// Replace the Op regions.
  void setOpRegions(mlir::ArrayAttr opRegions);

  /// Attributes.
  static void buildDefaultValuedAttrs(mlir::OpBuilder &builder,
                                      mlir::OperationState &result);

  static inline llvm::StringRef getOpAttrDictAttrName() {
    return "op_attrs";
  }
  static inline llvm::StringRef getOpTraitsAttrName() {
    return "traits";
  }
  static inline llvm::StringRef getOpRegionsAttrName() {
    return "regions";
  }
  static inline llvm::StringRef getIsTerminatorAttrName() {
    return "is_terminator";
  }
  static inline llvm::StringRef getIsCommutativeAttrName() {
    return "is_commutative";
  }
  static inline llvm::StringRef getIsIsolatedFromAboveAttrName() {
    return "is_isolated_from_above";
  }
};

/// The TypeOp allows dialects to define custom types by composing Attributes.
///
/// dmc.Type @My2DArray<#dmc.Type, #dmc.UI32, #dmc.UI32>
///
/// This will generate a type: u.My2DArray<i32, 4, 5>.
///
/// TODO support for type constraints, named parameters, type conversion,
/// imcomplete types with ?.
class TypeOp
    : public mlir::Op<TypeOp,
                      mlir::OpTrait::ZeroOperands, mlir::OpTrait::ZeroResult,
                      mlir::OpTrait::IsIsolatedFromAbove,
                      mlir::OpTrait::HasParent<DialectOp>::Impl,
                      mlir::SymbolOpInterface::Trait,
                      mlir::dmc::ParameterList::Trait,
                      mlir::dmc::ReparseOpInterface::Trait> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "dmc.Type"; }

  static void build(mlir::OpBuilder &builder, mlir::OperationState &result,
                    llvm::StringRef name,
                    llvm::ArrayRef<mlir::Attribute> parameters);

  /// Operation hooks.
  static mlir::ParseResult parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result);
  void print(mlir::OpAsmPrinter &printer);

  /// Reparse nested types and attributes.
  mlir::ParseResult reparse();
};

class AttributeOp
    : public mlir::Op<AttributeOp,
                      mlir::OpTrait::ZeroOperands, mlir::OpTrait::ZeroResult,
                      mlir::OpTrait::IsIsolatedFromAbove,
                      mlir::OpTrait::HasParent<DialectOp>::Impl,
                      mlir::SymbolOpInterface::Trait,
                      mlir::dmc::ParameterList::Trait,
                      mlir::dmc::ReparseOpInterface::Trait> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "dmc.Attr"; }

  static void build(mlir::OpBuilder &builder, mlir::OperationState &result,
                    llvm::StringRef name,
                    llvm::ArrayRef<mlir::Attribute> parameters);

  /// Operation hooks.
  static mlir::ParseResult parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result);
  void print(mlir::OpAsmPrinter &printer);

  /// Reparse nested types and attributes.
  mlir::ParseResult reparse();
};

/// An alias to a type or attribute.
class AliasOp
    : public mlir::Op<AliasOp,
                      mlir::OpTrait::ZeroOperands, mlir::OpTrait::ZeroResult,
                      mlir::OpTrait::IsIsolatedFromAbove,
                      mlir::OpTrait::HasParent<DialectOp>::Impl,
                      mlir::SymbolOpInterface::Trait,
                      mlir::dmc::ReparseOpInterface::Trait> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "dmc.Alias"; }

  static void build(mlir::OpBuilder &builder, mlir::OperationState &result,
                    llvm::StringRef name, mlir::Type type);
  static void build(mlir::OpBuilder &builder, mlir::OperationState &result,
                    llvm::StringRef name, mlir::Attribute attr);

  /// Operation hooks.
  static mlir::ParseResult parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result);
  void print(mlir::OpAsmPrinter &printer);

  /// Either one of the type or attribute must be present.
  mlir::LogicalResult verify();

  /// Reparse the type or attribute.
  mlir::ParseResult reparse();

  /// Getters.
  mlir::Type getAliasedType();
  mlir::Attribute getAliasedAttr();

private:
  static llvm::StringRef getAliasedTypeAttrName() { return "type"; }
  static llvm::StringRef getAliasedAttributeAttrName() { return "attr"; }
};

} // end namespace dmc
