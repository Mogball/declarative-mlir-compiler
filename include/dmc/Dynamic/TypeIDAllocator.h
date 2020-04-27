#pragma once

#include <mlir/Support/TypeID.h>

namespace dmc {

/// MLIR relies on static type IDs of classes, such as Dialect, Type,
/// and Attribute, to manage objects. Since we are dynamically creating
/// objects, we need to dynamically allocate TypeIDs.
class TypeIDAllocator {
public:
  virtual mlir::TypeID allocateID() = 0;
};

TypeIDAllocator *getFixedTypeIDAllocator();

} // end namespace dmc
