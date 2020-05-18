#include "dmc/Traits/OpTrait.h"
#include "dmc/Traits/Registry.h"

#include <mlir/IR/Diagnostics.h>

using namespace mlir;

namespace dmc {

namespace detail {
struct OpTraitStorage : public AttributeStorage {
  using KeyTy = std::pair<mlir::StringAttr, mlir::ArrayAttr>;

  explicit OpTraitStorage(const KeyTy &key)
      : name{key.first},
        params{key.second} {}

  bool operator==(const KeyTy &key) const {
    return key.first == name && key.second == params;
  }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  static OpTraitStorage *construct(AttributeStorageAllocator &alloc,
                                   const KeyTy &key) {
    return new (alloc.allocate<OpTraitStorage>()) OpTraitStorage{key};
  }

  mlir::StringAttr name;
  mlir::ArrayAttr params;
};

struct OpTraitsStorage : public AttributeStorage {
  using KeyTy = mlir::ArrayAttr;

  explicit OpTraitsStorage(KeyTy key) : traits{key} {}
  bool operator==(const KeyTy &key) const { return key == traits; }
  static llvm::hash_code hashKey(const KeyTy &key) { return hash_value(key); }

  static OpTraitsStorage *construct(AttributeStorageAllocator &alloc,
                                    KeyTy key) {
    return new (alloc.allocate<OpTraitsStorage>()) OpTraitsStorage{key};
  }

  KeyTy traits;
};
} // end namespace detail

OpTraitAttr OpTraitAttr::get(mlir::StringAttr nameAttr,
                             mlir::ArrayAttr paramAttr) {
  return Base::get(nameAttr.getContext(), TraitAttr::OpTrait, nameAttr,
                   paramAttr);
}

OpTraitAttr OpTraitAttr::getChecked(
    mlir::Location loc, mlir::StringAttr nameAttr, mlir::ArrayAttr paramAttr) {
  return Base::getChecked(loc, TraitAttr::OpTrait, nameAttr, paramAttr);
}

LogicalResult OpTraitAttr::verifyConstructionInvariants(
    mlir::Location loc, mlir::StringAttr nameAttr, mlir::ArrayAttr paramAttr) {
  if (!nameAttr)
    return emitError(loc) << "op trait name cannot be null";
  if (!paramAttr)
    return emitError(loc) << "op trait parameter list cannot be null";
  /// Check that the op trait exists.
  auto *registry = loc.getContext()->getRegisteredDialect<TraitRegistry>();
  assert(registry && "TraitRegistry dialect was not registered");
  if (!registry->lookupTrait(nameAttr.getValue()))
    return emitError(loc) << "op trait " << nameAttr << " not found";
  return success();
}

StringRef OpTraitAttr::getName() {
  return getImpl()->name.getValue();
}

ArrayRef<Attribute> OpTraitAttr::getParameters() {
  return getImpl()->params.getValue();
}

OpTraitsAttr OpTraitsAttr::get(mlir::ArrayAttr traits){
  return Base::get(traits.getContext(), TraitAttr::OpTraits, traits);
}

OpTraitsAttr OpTraitsAttr::getChecked(Location loc, mlir::ArrayAttr traits) {
  return Base::getChecked(loc, TraitAttr::OpTraits, traits);
}

LogicalResult OpTraitsAttr::verifyConstructionInvariants(
    Location loc, mlir::ArrayAttr traits) {
  if (!traits)
    return emitError(loc) << "op traits list cannot be null";
  for (auto trait : traits) {
    if (!trait.isa<OpTraitAttr>())
      return emitError(loc) << "op traits list must only contain op traits";
  }
  return success();
}

ArrayRef<Attribute> OpTraitsAttr::getUnderlyingValue() {
  return getImpl()->traits.getValue();
}

} // end namespace dmc
