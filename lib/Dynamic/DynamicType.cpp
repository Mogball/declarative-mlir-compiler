#include "dmc/Dynamic/DynamicType.h"
#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Spec/SpecAttrImplementation.h"

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

DynamicTypeImpl::DynamicTypeImpl(DynamicContext *ctx, StringRef name,
                                 ArrayRef<Attribute> paramSpec)
    : DynamicObject{ctx},
      name{name},
      paramSpec{paramSpec} {}

Type DynamicTypeImpl::parseType(Location loc, DialectAsmParser &parser) {
  std::vector<Attribute> params;
  params.reserve(paramSpec.size());
  if (!parser.parseOptionalLess()) {
    do {
      Attribute attr;
      if (parser.parseAttribute(attr))
        return Type{};
      params.push_back(attr);
    } while (!parser.parseOptionalComma());
    if (parser.parseGreater())
      return Type{};
  }
  return DynamicType::getChecked(loc, this, params);
}

void DynamicTypeImpl::printType(Type type, DialectAsmPrinter &printer) {
  auto dynTy = type.cast<DynamicType>();
  printer << name;
  if (!paramSpec.empty()) {
    auto it = std::begin(paramSpec);
    printer << '<' << (*it++);
    for (auto e = std::end(paramSpec); it != e; ++it)
      printer << ',' << (*it);
    printer << '>';
  }
}

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
  if (llvm::size(impl->paramSpec) != llvm::size(params))
    return emitError(loc) << "Dynamic type construction failed: expected "
         << llvm::size(impl->paramSpec) << " parameters but got "
         << llvm::size(params);
  /// Verify that the provided parameters satisfy the dynamic type spec.
  unsigned idx = 0;
  for (auto [spec, param] : llvm::zip(impl->paramSpec, params)) {
    if (failed(SpecAttrs::delegateVerify(spec, param)))
      return emitError(loc) << "Dynamic type construction failed: parameter #"
          << idx << " expected " << spec << " but got " << param;
    ++idx;
  }
  return success();
}

DynamicTypeImpl *DynamicType::getTypeImpl() {
  return getImpl()->impl;
}

ArrayRef<Attribute> DynamicType::getParams() {
  return getImpl()->params;
}

} // end namespace dmc
