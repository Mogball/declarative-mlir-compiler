#pragma once

#include "Metadata.h"
#include "DynamicObject.h"
#include "dmc/Kind.h"
#include "dmc/Spec/ParameterList.h"

#include <mlir/IR/DialectImplementation.h>

namespace dmc {

/// Forward declarations.
class DynamicDialect;

namespace detail{
struct DynamicTypeStorage;
} // end namespace detail

/// DynamicType underlying class. The class stores type class functions like
/// the parser, printer, and conversions. Each dynamic Type instance holds a
/// reference to an instance of this class.
class DynamicTypeImpl : public DynamicObject, public TypeMetadata {
public:
  /// Create a dynamic type with the provided name and parameter spec.
  explicit DynamicTypeImpl(DynamicDialect *dialect, llvm::StringRef name,
                           NamedParameterRange paramSpec);

  /// Getters.
  inline DynamicDialect *getDialect() { return dialect; }
  inline auto getParamSpec()  { return paramSpec; }

  /// Delegate parser and printer.
  mlir::Type parseType(mlir::Location loc, mlir::DialectAsmParser &parser);
  void printType(mlir::Type type, mlir::DialectAsmPrinter &printer);

private:
  /// The dialect to which this type belongs.
  DynamicDialect *dialect;
  /// The parameters are defined by Attribute constraints. The Attribute
  /// instances must be Spec attributes.
  NamedParameterRange paramSpec;

  friend class DynamicType;
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
  /// Verify that the parameter attributes are valid.
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
