#pragma once

#include "SpecTypeImplementation.h"

namespace dmc {

namespace detail {

/// Storage for SpecTypes parameterized by a width.
struct WidthStorage : public mlir::TypeStorage {
  /// Use width as key.
  using KeyTy = unsigned;

  explicit WidthStorage(KeyTy key);
  bool operator==(const KeyTy &key) const;
  static llvm::hash_code hashKey(const KeyTy &key);
  static WidthStorage *construct(mlir::TypeStorageAllocator &alloc,
                                 const KeyTy &key);

  KeyTy width;
};

} // end namespace detail

} // end namespace dmc
