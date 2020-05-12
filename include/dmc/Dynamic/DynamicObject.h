#pragma once

#include <mlir/Support/TypeID.h>

namespace dmc {

/// Forward declarations.
class DynamicContext;

/// TypeIDs are not associated with a class type but are assigned to an instance
/// of a dynamic object that mocks an otherwise statically known class.
class DynamicObject {
public:
  explicit DynamicObject(DynamicContext *ctx);

  inline DynamicContext *getDynContext() const { return ctx; }
  inline mlir::TypeID getTypeID() { return typeId; }

private:
  DynamicContext *ctx;
  mlir::TypeID typeId;
};

} // end namespace dmc
