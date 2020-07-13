#pragma once

#include "Kinds.h"
#include "dmc/Dynamic/DynamicOperation.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/OpDefinition.h>

/// Bind common OpTraits into DynamicTraits
namespace dmc {

template <template <typename ConcreteType> class BaseTrait>
struct BindTrait : public DynamicTrait {
  mlir::LogicalResult verifyOp(mlir::Operation *op) const override {
    // Provide dummy template arg
    return BaseTrait<DynamicTrait>::verifyTrait(op);
  }

  mlir::AbstractOperation::OperationProperties
  getTraitProperties() const override {
    return BaseTrait<DynamicTrait>::getTraitProperties();
  }
};
template <typename Arg>
class BindArgTrait : public DynamicTrait {
protected:
  using ArgTy = Arg;
  using TraitImpl = typename
    std::add_pointer<mlir::LogicalResult(mlir::Operation *, ArgTy)>::type;

public:
  BindArgTrait(TraitImpl impl, Arg arg) : impl{impl}, arg{arg} {};

  mlir::LogicalResult verifyOp(mlir::Operation *op) const override {
    return impl(op, arg);
  }

private:
  TraitImpl impl;
  ArgTy arg;
};

/// Simple traits can be rebound.
struct IsTerminator
    : public BindTrait<mlir::OpTrait::IsTerminator> {
  static llvm::StringRef getName() { return "IsTerminator"; }
};
struct IsCommutative
    : public BindTrait<mlir::OpTrait::IsCommutative> {
  static llvm::StringRef getName() { return "IsCommutative"; }
};
struct IsIsolatedFromAbove
    : public BindTrait<mlir::OpTrait::IsIsolatedFromAbove> {
  static llvm::StringRef getName() { return "IsIsolatedFromAbove"; }
};

struct OperandsAreFloatLike
    : public BindTrait<mlir::OpTrait::OperandsAreFloatLike> {
  static llvm::StringRef getName() { return "OperandsAreFloatLike"; }
};
struct OperandsAreSignlessIntegerLike
    : public BindTrait<mlir::OpTrait::OperandsAreSignlessIntegerLike> {
  static llvm::StringRef getName() { return "OperandsAreSignlessIntegerLike"; }
};
struct ResultsAreBoolLike
    : public BindTrait<mlir::OpTrait::ResultsAreBoolLike> {
  static llvm::StringRef getName() { return "ResultsAreBoolLike"; }
};
struct ResultsAreFloatLike
    : public BindTrait<mlir::OpTrait::ResultsAreFloatLike> {
  static llvm::StringRef getName() { return "ResultsAreFloatLike"; }
};
struct ResultsAreSignlessIntegerLike
    : public BindTrait<mlir::OpTrait::ResultsAreSignlessIntegerLike> {
  static llvm::StringRef getName() { return "ResultsAreSignlessIntegerLike"; }
};

struct SameOperandsShape
    : public BindTrait<mlir::OpTrait::SameOperandsShape> {
  static llvm::StringRef getName() { return "SameOperandsShape"; }
};
struct SameOperandsAndResultShape
    : public BindTrait<mlir::OpTrait::SameOperandsAndResultShape> {
  static llvm::StringRef getName() { return "SameOperandsAndResultShape"; }
};
struct SameOperandsElementType
    : public BindTrait<mlir::OpTrait::SameOperandsElementType> {
  static llvm::StringRef getName() { return "SameOperandsElementType"; }
};
struct SameOperandsAndResultElementType
    : public BindTrait<mlir::OpTrait::SameOperandsAndResultElementType> {
  static llvm::StringRef getName() { return "SameOperandsAndResultElementType"; }
};
struct SameOperandsAndResultType
    : public BindTrait<mlir::OpTrait::SameOperandsAndResultType> {
  static llvm::StringRef getName() { return "SameOperandsAndResultType"; }
};
struct SameTypeOperands
    : public BindTrait<mlir::OpTrait::SameTypeOperands> {
  static llvm::StringRef getName() { return "SameTypeOperands"; }
};

/// Stateful traits require constructors.
struct NOperands : public BindArgTrait<unsigned> {
  static llvm::StringRef getName() { return "NOperands"; }

  explicit NOperands(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyNOperands, num) {}
};
struct AtLeastNOperands : public BindArgTrait<unsigned> {
  static llvm::StringRef getName() { return "AtLeastNOperands"; }

  explicit AtLeastNOperands(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyAtLeastNOperands, num) {}
};
struct NRegions : public BindArgTrait<unsigned> {
  static llvm::StringRef getName() { return "NRegions"; }

  explicit NRegions(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyNRegions, num) {}
};
struct AtLeastNRegions : public BindArgTrait<unsigned> {
  static llvm::StringRef getName() { return "AtLeastNRegions"; }

  explicit AtLeastNRegions(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyAtLeastNRegions, num) {}
};
struct NResults : public BindArgTrait<unsigned> {
  static llvm::StringRef getName() { return "NResults"; }

  explicit NResults(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyNResults, num) {}
};
struct AtLeastNResults : public BindArgTrait<unsigned> {
  static llvm::StringRef getName() { return "AtLeastNResults"; }

  explicit AtLeastNResults(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyAtLeastNResults, num) {}
};
struct NSuccessors : public BindArgTrait<unsigned> {
  static llvm::StringRef getName() { return "NSuccessors"; }

  explicit NSuccessors(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyNSuccessors, num) {}
};
struct AtLeastNSuccessors : public BindArgTrait<unsigned> {
  static llvm::StringRef getName() { return "AtLeastNSuccessors"; }

  explicit AtLeastNSuccessors(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyAtLeastNSuccessors, num) {}
};

/// Non-trivial stateful traits.
class HasParent : public DynamicTrait {
public:
  static llvm::StringRef getName() { return "HasParent"; }
  explicit HasParent(llvm::StringRef parentName) : parentName{parentName} {}
  mlir::LogicalResult verifyOp(mlir::Operation *op) const override;

private:
  llvm::StringRef parentName;
};

class SingleBlockImplicitTerminator : public DynamicTrait {
public:
  static llvm::StringRef getName() { return "SingleBlockImplicitTerminator"; }
  explicit SingleBlockImplicitTerminator(llvm::StringRef terminatorName)
      : terminatorName{terminatorName} {}
  mlir::LogicalResult verifyOp(mlir::Operation *op) const override;

private:
  llvm::StringRef terminatorName;
};

class MemoryAlloc : public DynamicTrait {
public:
  static llvm::StringRef getName() { return "MemoryAlloc"; }
};

class MemoryFree : public DynamicTrait {
public:
  static llvm::StringRef getName() { return "MemoryFree"; }
};

class MemoryRead : public DynamicTrait {
public:
  static llvm::StringRef getName() { return "MemoryRead"; }
};

class MemoryWrite : public DynamicTrait {
public:
  static llvm::StringRef getName() { return "MemoryWrite"; }
};

class ValueMemoryEffect : public DynamicTrait {
public:
  explicit ValueMemoryEffect(std::vector<llvm::StringRef> targets)
      : targets{std::move(targets)} {}

  auto &getTargets() const { return targets; }

private:
  std::vector<llvm::StringRef> targets;
};

class Alloc : public ValueMemoryEffect {
public:
  static llvm::StringRef getName() { return "Alloc"; }
  explicit Alloc(mlir::Attribute targets);
};

class Free : public ValueMemoryEffect {
public:
  static llvm::StringRef getName() { return "Free"; }
  explicit Free(mlir::Attribute targets);
};

class ReadFrom : public ValueMemoryEffect {
public:
  static llvm::StringRef getName() { return "ReadFrom"; }
  explicit ReadFrom(mlir::Attribute targets);
};

class WriteTo : public ValueMemoryEffect {
public:
  static llvm::StringRef getName() { return "WriteTo"; }
  explicit WriteTo(mlir::Attribute targets);
};

class NoSideEffects : public DynamicTrait {
public:
  static llvm::StringRef getName() { return "NoSideEffects"; }
};

class LoopLike : public DynamicTrait {
public:
  static llvm::StringRef getName() { return "LoopLike"; }
  explicit LoopLike(llvm::StringRef region, llvm::StringRef definedOutsideFcn)
      : region{region}, definedOutsideFcn{definedOutsideFcn} {}

  mlir::Region &getLoopRegion(DynamicOperation *impl, mlir::Operation *op);
  bool isDefinedOutside(DynamicOperation *impl, mlir::Operation *op,
                        mlir::Value value);

private:
  llvm::StringRef region, definedOutsideFcn;
};

/// TODO Some standard traits not rebound (complexity/API restrictions):
/// - AutomaticAllocationScope
/// - ConstantLike

} // end namespace dmc
