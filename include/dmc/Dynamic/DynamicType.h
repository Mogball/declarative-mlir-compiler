#pragma once

#include "DynamicObject.h"
#include "dmc/Kind.h"

namespace dmc {

namespace detail{
struct DynamicTypeStorage;
}

/// DynamicType underlying class. The class stores type class functions like
/// the parser, printer, and conversions. Each dynamic Type instance holds a
/// reference to an instance of this class.
class DynamicTypeImpl : public DynamicObject {
public:
  explicit DynamicTypeImpl(DynamicContext *ctx);
};

/// DynamicType class. Stores parameters according to DynamicTypeImpl.
class DynamicType : public mlir::Type::TypeBase<DynamicType, mlir::Type,
                                                detail::DynamicTypeStorage> {
  /// Static casting between with DynamicType doesn't make sense so provide
  /// a single kind for casting to DynamicType.
  static constexpr auto DynamicTypeKind = dmc::Kind::FIRST_DYNAMIC_TYPE;

public:
  using Base::Base;

  /// Get a DynamicType with a backing DynamicTypeImpl and provided parameter
  /// values.
  static DynamicType get(DynamicTypeImpl *impl,
                         llvm::ArrayRef<mlir::Attribute> params);
  static DynamicType getChecked(mlir::Location loc, DynamicTypeImpl *impl,
                                llvm::ArrayRef<mlir::Attribute> params);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, DynamicTypeImpl *impl,
      llvm::ArrayRef<mlir::Attribute> params);

  /// Allow casting of Type to DynamicType.
  static bool kindof(unsigned kind) { return kind == DynamicTypeKind; }

  /// Getters.
  DynamicTypeImpl *getTypeImpl();
  llvm::ArrayRef<mlir::Attribute> getParams();
};

} // end namespace dmc
