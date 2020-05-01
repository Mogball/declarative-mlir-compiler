#include "dmc/Spec/SpecAttrs.h"
#include "dmc/Spec/Support.h"

#include <llvm/ADT/SmallPtrSet.h>

using namespace mlir;

namespace dmc {

namespace detail {

/// TypedAttrStorage implementation.
TypedAttrStorage::TypedAttrStorage(KeyTy key) : type{key} {}

bool TypedAttrStorage::operator==(const KeyTy &key) const { 
  return key == type; 
}

llvm::hash_code TypedAttrStorage::hashKey(const KeyTy &key) {
  return hash_value(key);
}

TypedAttrStorage *TypedAttrStorage::construct(
    AttributeStorageAllocator &alloc, const KeyTy &key) {
  return new (alloc.allocate<TypedAttrStorage>())
      TypedAttrStorage{key};
}

/// ConstantAttrStorage implementation.
struct ConstantAttrStorage : public AttributeStorage {
  using KeyTy = Attribute;

  explicit ConstantAttrStorage(KeyTy key) : attr{key} {}
  bool operator==(const KeyTy &key) const { return key == attr; }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return hash_value(key);
  }

  static ConstantAttrStorage *construct(
      AttributeStorageAllocator &alloc, const KeyTy &key) {
    return new (alloc.allocate<ConstantAttrStorage>())
        ConstantAttrStorage{key};
  }

  KeyTy attr;
};

/// AttrListStorage implementation.
struct AttrListStorage : public AttributeStorage {
  using KeyTy = ImmutableSortedList<Attribute>;
  
  explicit AttrListStorage(KeyTy key) : attrs{std::move(key)} {}
  bool operator==(const KeyTy &key) const { return key == attrs; }
  static llvm::hash_code hashKey(const KeyTy &key) { return key.hash(); }

  static AttrListStorage *construct(
      AttributeStorageAllocator &alloc, KeyTy key) {
    return new (alloc.allocate<AttrListStorage>())
      AttrListStorage{std::move(key)};
  }

  KeyTy attrs;
};

struct AttrComparator {
  bool operator()(Attribute lhs, Attribute rhs) const {
    return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
  }
};

} // end namespace detail

/// ConstantAttr implementation.
ConstantAttr ConstantAttr::get(Attribute attr) {
  return Base::get(attr.getContext(), SpecAttrs::Constant, attr);
}

ConstantAttr ConstantAttr::getChecked(Location loc, Attribute attr) {
  return Base::getChecked(loc, SpecAttrs::Constant, attr);
}

LogicalResult ConstantAttr::verifyConstructionInvariants(
    Location loc, Attribute attr) {
  if (!attr)
    return emitError(loc) << "Attribute cannot be null";
  return success();
}

LogicalResult ConstantAttr::verify(Attribute attr) {
  return success(attr == getImpl()->attr);
}

/// Helper functions.
namespace {

namespace impl {

LogicalResult verifyAttrList(Location loc, ArrayRef<Attribute> attrs) {
  if (attrs.empty())
    return emitError(loc) << "attribute list cannot be empty";
  llvm::SmallPtrSet<Attribute, 4> attrSet{std::begin(attrs), 
                                          std::end(attrs)};
  if (std::size(attrSet) != std::size(attrs))
    return emitError(loc) << "duplicate attributes passed";
  return success();
}
  
} // end namespace impl

auto getSortedAttrs(ArrayRef<Attribute> attrs) {
  return getSortedListOf<detail::AttrComparator>(attrs);
}

} // end anonymous namespace

/// AnyOfAttr implementation
AnyOfAttr AnyOfAttr::get(ArrayRef<Attribute> attrs) {
  auto *ctx = attrs.front().getContext();
  return Base::get(ctx, SpecAttrs::AnyOf, getSortedAttrs(attrs));
}

AnyOfAttr AnyOfAttr::getChecked(Location loc, ArrayRef<Attribute> attrs) {
  return Base::getChecked(loc, SpecAttrs::AnyOf, getSortedAttrs(attrs));
}

LogicalResult AnyOfAttr::verifyConstructionInvariants(
    Location loc, ArrayRef<Attribute> attrs) {
  return impl::verifyAttrList(loc, attrs);
}

LogicalResult AnyOfAttr::verify(Attribute attr) {
  for (auto baseAttr : getImpl()->attrs) {
    if (SpecAttrs::is(baseAttr) &&
        succeeded(SpecAttrs::delegateVerify(baseAttr, attr)))
      return success();
    else if (baseAttr == attr)
      return success();
  }
  return failure();
}

/// AllOfAttr implementation
AllOfAttr AllOfAttr::get(ArrayRef<Attribute> attrs) {
  auto *ctx = attrs.front().getContext();
  return Base::get(ctx, SpecAttrs::AllOf, getSortedAttrs(attrs));
}

AllOfAttr AllOfAttr::getChecked(Location loc, ArrayRef<Attribute> attrs) {
  return Base::getChecked(loc, SpecAttrs::AllOf, getSortedAttrs(attrs));
}

LogicalResult AllOfAttr::verifyConstructionInvariants(
    Location loc, ArrayRef<Attribute> attrs) {
  return impl::verifyAttrList(loc, attrs); 
}

LogicalResult AllOfAttr::verify(Attribute attr) {
  for (auto baseAttr : getImpl()->attrs) {
    if (SpecAttrs::is(baseAttr) &&
        failed(SpecAttrs::delegateVerify(baseAttr, attr)))
      return failure();
    else if (baseAttr != attr)
      return failure();
  }
  return success();
}

} // end namespace dmc
