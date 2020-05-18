#pragma once

#include "dmc/Dynamic/DynamicOperation.h"

#include <llvm/ADT/StringMap.h>
#include <mlir/IR/Dialect.h>

namespace dmc {

/// The trait registry is a Dialect so that it can be stored inside the
/// MLIRContext for later lookup.
class TraitRegistry : public mlir::Dialect {
public:
  /// TODO stateful traits, e.g. @MyTrait<arg1, arg2>.
  using Trait = std::unique_ptr<DynamicTrait>;
  using TraitConstructor = std::function<Trait()>;

  explicit TraitRegistry(mlir::MLIRContext *ctx);
  static llvm::StringRef getDialectNamespace() { return "trait";  }

  /// Register a stateless trait.
  void registerTrait(llvm::StringRef name, TraitConstructor getter);
  /// Lookup a stateless trait.
  Trait lookupTrait(llvm::StringRef name);

  /// OpTrait attribute parsing and printing.
  mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
                                 mlir::Type type) const override;
  void printAttribute(mlir::Attribute attr,
                      mlir::DialectAsmPrinter &printer) const override;

private:
  llvm::StringMap<TraitConstructor> traitRegistry;

  /// Register a static trait.
  template <typename TraitT>
  void registerTrait() {
    registerTrait(TraitT::getName(),
        [] { return std::make_unique<TraitT>(); });
  }
  /// Register multiple static traits.
  template <typename... TraitTys>
  void registerTraits() {
    (void) std::initializer_list<int>{0, (registerTrait<TraitTys>(), 0)...};
  }
};

/// Of-out-line definitions.

} // end namespace dmc
