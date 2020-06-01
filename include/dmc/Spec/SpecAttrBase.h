#pragma once

#include <mlir/IR/Attributes.h>

/// Make these definitions public so that other constraint kinds can use them.
namespace dmc {
namespace detail {

/// OneAttrStorage implementation. Store one attribute.
struct OneAttrStorage : public mlir::AttributeStorage {
  using KeyTy = mlir::Attribute;

  inline explicit OneAttrStorage(KeyTy key) : attr{key} {}
  inline bool operator==(const KeyTy &key) const { return key == attr; }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return hash_value(key);
  }

  static OneAttrStorage *construct(mlir::AttributeStorageAllocator &alloc,
                                   const KeyTy &key) {
    return new (alloc.allocate<OneAttrStorage>()) OneAttrStorage{key};
  }

  KeyTy attr;
};

} // end namespace detail
} // end namespace dmc
