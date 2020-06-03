#include "dmc/Dynamic/DynamicAttribute.h"
#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Spec/SpecAttrImplementation.h"

#include <mlir/IR/Location.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/StandardTypes.h>

using namespace mlir;

namespace dmc {

/// DynamicAttributeStorage stores a reference to the DynamicAttribute
/// class descriptor.
namespace detail {
struct DynamicAttributeStorage : public AttributeStorage {
  using KeyTy = std::pair<DynamicAttributeImpl *, ArrayRef<Attribute>>;

  explicit DynamicAttributeStorage(DynamicAttributeImpl *impl,
                                   ArrayRef<Attribute> params)
      : impl{impl}, params{params} {}

  bool operator==(const KeyTy &key) const {
    return impl == key.first && params == key.second;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  static DynamicAttributeStorage *construct(AttributeStorageAllocator &alloc,
                                            const KeyTy &key) {
    return new (alloc.allocate<DynamicAttributeStorage>())
        DynamicAttributeStorage{key.first, alloc.copyInto(key.second)};
  }

  DynamicAttributeImpl *impl;
  ArrayRef<Attribute> params;
};
} // end namespace detail

DynamicAttributeImpl::DynamicAttributeImpl(
    DynamicDialect *dialect, StringRef name, ArrayRef<Attribute> paramSpec)
    : DynamicObject{dialect->getDynContext()},
      AttributeMetadata{name, llvm::None, Type{}},
      dialect{dialect},
      paramSpec{paramSpec} {}

Attribute DynamicAttributeImpl::parseAttribute(Location loc,
                                               DialectAsmParser &parser) {
  std::vector<Attribute> params;
  params.reserve(std::size(paramSpec));
  if (!parser.parseOptionalLess()) {
    do {
      Attribute attr;
      if (parser.parseAttribute(attr))
        return {};
      params.push_back(attr);
    } while (!parser.parseOptionalComma());
    if (parser.parseGreater())
      return {};
  }
  return DynamicAttribute::getChecked(loc, this, params);
}

void DynamicAttributeImpl::printAttribute(Attribute attr,
                                          DialectAsmPrinter &printer) {
  auto dynAttr = attr.cast<DynamicAttribute>();
  printer << getName();
  auto params = dynAttr.getParams();
  if (!params.empty()) {
    auto it = std::begin(params);
    printer << '<' << (*it++);
    for (auto e = std::end(params); it != e; ++it)
      printer << ',' << (*it);
    printer << '>';
  }
}

/// Since dynamic attributes are not registered with a Dialect or the MLIR
/// context, we need to directly call the Attribute uniquer.
DynamicAttribute DynamicAttribute::get(DynamicAttributeImpl *impl,
                                       ArrayRef<Attribute> params) {
  auto *ctx = impl->getDynContext()->getContext();
  return ctx->getAttributeUniquer().get<Base::ImplType>(
      [impl, ctx](AttributeStorage *storage) {
        storage->initializeDialect(*impl->getDialect());
        /// If a type was not provided, default to NoneType.
        if (!storage->getType())
          storage->setType(NoneType::get(ctx));
      },
      DynamicAttributeKind, impl, params);
}

DynamicAttribute DynamicAttribute::getChecked(
    Location loc, DynamicAttributeImpl *impl, ArrayRef<Attribute> params) {
  if (failed(verifyConstructionInvariants(loc, impl, params)))
    return {};
  return get(impl, params);
}

LogicalResult DynamicAttribute::verifyConstructionInvariants(
  Location loc, DynamicAttributeImpl *impl, ArrayRef<Attribute> params) {
  if (impl == nullptr)
    return emitError(loc) << "Null DynamicAttributeImpl";
  if (llvm::size(impl->paramSpec) != llvm::size(params))
    return emitError(loc) << "attribute construction failed: expected "
        << llvm::size(impl->paramSpec) << " parameters but got "
        << llvm::size(params);
  unsigned idx = 0;
  for (auto [spec, param] : llvm::zip(impl->paramSpec, params)) {
    if (failed(SpecAttrs::delegateVerify(spec, param)))
      return emitError(loc) << "attribute construction failed: parameter #"
           << idx << " expected " << spec << " but got " << param;
    ++idx;
  }
  return success();
}

DynamicAttributeImpl *DynamicAttribute::getAttrImpl() {
  return getImpl()->impl;
}

ArrayRef<Attribute> DynamicAttribute::getParams() {
  return getImpl()->params;
}

} // end namespac dmc
