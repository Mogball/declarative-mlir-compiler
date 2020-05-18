#include "dmc/Spec/OpTrait.h"
#include "dmc/Traits/Registry.h"

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

  static OpTraitStorage *construct(
      AttributeStorageAllocator &alloc, const KeyTy &key) {
    return new (alloc.allocate<OpTraitStorage>()) OpTraitStorage{key};
  }

  mlir::StringAttr name;
  mlir::ArrayAttr params;
};
} // end namespace detail

OpTrait OpTrait::get(mlir::StringAttr nameAttr, mlir::ArrayAttr paramAttr) {
  return Base::get(nameAttr.getContext(), SpecAttrs::OpTrait, nameAttr,
                   paramAttr);
}

OpTrait OpTrait::getChecked(mlir::Location loc, mlir::StringAttr nameAttr,
                            mlir::ArrayAttr paramAttr) {
  return Base::getChecked(loc, SpecAttrs::OpTrait, nameAttr, paramAttr);
}

LogicalResult OpTrait::verifyConstructionInvariants(
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

StringRef OpTrait::getName() {
  return getImpl()->name.getValue();
}

ArrayRef<Attribute> OpTrait::getParameters() {
  return getImpl()->params.getValue();
}

} // end namespace dmc
