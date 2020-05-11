#include "dmc/Dynamic/DynamicType.h"
#include "dmc/Dynamic/DynamicContext.h"

#include <mlir/IR/Location.h>
#include <mlir/IR/Diagnostics.h>

using namespace mlir;

namespace dmc {

/// DynamicTypeStorage stores a reference to the backing DynamicTypeImpl,
/// which conveniently acts as a discriminator between different DynamicType
/// "classes", and the parameter values.
namespace detail {
struct DynamicTypeStorage : public TypeStorage {
  /// Compound key with the Impl instance and the parameter values.
  using KeyTy = std::pair<DynamicTypeImpl *, ArrayRef<Attribute>>;

  explicit DynamicTypeStorage(const KeyTy &key)
      : impl{key.first},
        params{std::begin(key.second), std::end(key.second)} {}

  /// Compare implmentation pointer and parameter values.
  bool operator==(const KeyTy &key) const {
    return impl == key.first &&
        params.size() == key.second.size() &&
        std::equal(std::begin(params), std::end(params),
                   std::begin(key.second));
  }

  /// Hash combine the implementation pointer and the parameter values.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  /// Create the DynamicTypeStorage.
  static DynamicTypeStorage *construct(TypeStorageAllocator &alloc,
                                       const KeyTy &key) {
    return new (alloc.allocate<DynamicTypeStorage>()) DynamicTypeStorage{key};
  }

  /// Pointer to implmentation.
  DynamicTypeImpl *impl;
  /// Store the parameters.
  std::vector<Attribute> params;
};
} // end namespace detail

DynamicType DynamicType::get(DynamicTypeImpl *impl,
                             ArrayRef<Attribute> params) {
  return Base::get(
      impl->getDynContext()->getContext(), DynamicTypeKind, impl, params);
}

DynamicType DynamicType::getChecked(Location loc, DynamicTypeImpl *impl,
                                    ArrayRef<Attribute> params) {
  return Base::getChecked(loc, DynamicTypeKind, impl, params);
}

LogicalResult DynamicType::verifyConstructionInvariants(
    Location loc, DynamicTypeImpl *impl, ArrayRef<Attribute> params) {
  if (impl == nullptr)
    return emitError(loc) << "Null DynamicTypeImpl";
  return success();
}

DynamicTypeImpl *DynamicType::getTypeImpl() {
  return getImpl()->impl;
}

ArrayRef<Attribute> DynamicType::getParams() {
  return getImpl()->params;
}

} // end namespace dmc
