#pragma once

#include "dmc/Dynamic/DynamicOperation.h"

#include <llvm/ADT/StringMap.h>
#include <mlir/IR/Dialect.h>

namespace dmc {

/// The trait registry is a Dialect so that it can be stored inside the
/// MLIRContext for later lookup.
class TraitRegistry : public mlir::Dialect {
public:
  using Trait = std::unique_ptr<DynamicTrait>;
  using TraitConstructor = std::function<Trait>;

  explicit TraitRegistry(mlir::MLIRContext *ctx);
  static llvm::StringRef getDialectNamespace() { return "trait";  }

  bool registerTrait(llvm::StringRef name, TraitConstructor getter);
  Trait lookupTrait(llvm::StringRef name);

private:
  llvm::StringMap<TraitConstructor> traitRegistry;
};

} // end namespace dmc
