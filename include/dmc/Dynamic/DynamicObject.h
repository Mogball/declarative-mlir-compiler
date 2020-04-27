#pragma once

#include <mlir/Support/TypeID.h>

namespace dmc {

/// Forward declarations.
class DynamicContext;

/// TypeIDs are not associated with a class type but are assigned to an instance
/// of a dynamic object that mocks an otherwise statically known class.
class DynamicObject {
public:
  DynamicObject(DynamicContext *ctx);

  inline mlir::TypeID getTypeID() { return typeId; }

private:
  mlir::TypeID typeId;
};

} // end namespace dmc
