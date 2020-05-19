#pragma once

#include "dmc/Dynamic/DynamicOperation.h"

#include <llvm/ADT/StringMap.h>
#include <mlir/IR/Dialect.h>

namespace dmc {

using Trait = std::unique_ptr<DynamicTrait>;

/// The trait constructor leverages MLIR's attribute system to store generic
/// values to pass to a "trait constructor". This is used to generically create
/// parameterized traits, such as @NSuccessors<2>.
class TraitConstructor {
public:
  using ArgsT = llvm::ArrayRef<mlir::Attribute>;

  /// Create the constructor with a signature verifier and a call function.
  template <typename VerifyFn, typename CallFn>
  TraitConstructor(VerifyFn verifyFunc, CallFn callFunc)
      : verifyFunc{verifyFunc}, callFunc{callFunc} {}

  /// Convertible from nullptr and to bool.
  TraitConstructor(std::nullptr_t) {}
  operator bool() const { return verifyFunc && callFunc; }

  /// Delegate to internal functions.
  inline auto call(ArgsT args) { return callFunc(args); }
  inline auto verify(mlir::Location loc, ArgsT args) {
    return verifyFunc(loc, args);
  }

private:
  /// The internal functions.
  std::function<mlir::LogicalResult(mlir::Location loc, ArgsT)> verifyFunc;
  std::function<Trait(ArgsT)> callFunc;
};

/// The trait registry is a Dialect so that it can be stored inside the
/// MLIRContext for later lookup.
class TraitRegistry : public mlir::Dialect {
public:
  explicit TraitRegistry(mlir::MLIRContext *ctx);
  static llvm::StringRef getDialectNamespace() { return "trait";  }

  /// Register a trait constructor.
  void registerTrait(llvm::StringRef name, TraitConstructor &&getter);
  /// Lookup a trait constructor.
  TraitConstructor lookupTrait(llvm::StringRef name);

  /// OpTrait attribute parsing and printing.
  mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
                                 mlir::Type type) const override;
  void printAttribute(mlir::Attribute attr,
                      mlir::DialectAsmPrinter &printer) const override;

private:
  llvm::StringMap<TraitConstructor> traitRegistry;
};

/// Of-out-line definitions.

} // end namespace dmc
