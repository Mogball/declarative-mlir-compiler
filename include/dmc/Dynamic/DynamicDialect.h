#pragma once

#include "DynamicObject.h"
#include "dmc/Spec/ParameterList.h"

#include <mlir/IR/Dialect.h>

namespace dmc {

/// Forward declarations.
class DynamicContext;
class DynamicOperation;
class DynamicTypeImpl;
class DynamicAttributeImpl;
class TypeAlias;
class AttributeAlias;
class TypeMetadata;
class AttributeMetadata;

/// Dynamic dialect underlying class. This class hooks Dialect methods
/// into user-specified functions.
class DynamicDialect : public mlir::Dialect,
                       public DynamicObject {
public:
  ~DynamicDialect();
  DynamicDialect(llvm::StringRef name, DynamicContext *ctx);

  /// Create a new Op associated with this Dialect. Additional configs are
  /// added directly to the returned DynamicOperation before it is finalized.
  std::unique_ptr<DynamicOperation> createDynamicOp(llvm::StringRef name);

  /// Create a dynamic type with the given name and parameter spec and add
  /// it to the given dialect.
  mlir::LogicalResult createDynamicType(llvm::StringRef name,
                                        NamedParameterRange paramSpec);

  /// Create a dynamic attribute with the given name and parameter spec and
  /// add it to the given dialect.
  mlir::LogicalResult createDynamicAttr(llvm::StringRef name,
                                        NamedParameterRange paramSpec);

  /// Expose configuration methods.
  inline void allowUnknownOperations(bool allow) {
    Dialect::allowUnknownOperations(allow);
  }
  inline void allowUnknownTypes(bool allow) {
    Dialect::allowUnknownTypes(allow);
  }

  /// Printing and parsing for dynamic types.
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;

  /// Printing and parsing for dynamic attributes.
  mlir::Attribute
  parseAttribute(mlir::DialectAsmParser &parser,
                 mlir::Type type) const override;
  void printAttribute(mlir::Attribute attr,
                      mlir::DialectAsmPrinter &printer) const override;

  /// Register a DynamicOperation with this dialect so its config
  /// is stored for later use. The dialect takes ownership.
  mlir::LogicalResult registerDynamicOp(std::unique_ptr<DynamicOperation> op);
  /// Lookup the DynamicOperation belonging to an Operation. Returns null if
  /// not found.
  DynamicOperation *lookupOp(mlir::OperationName name) const;

  /// Register a DynamicType with the dialect. The dialect takes ownership.
  mlir::LogicalResult
  registerDynamicType(std::unique_ptr<DynamicTypeImpl> type);
  /// Lookup a DynamicType with the given name. Returns nullptr if none is
  /// found.
  DynamicTypeImpl *lookupType(llvm::StringRef name) const;

  /// Register a DynamicAttribute with the dialect. The dialect takes
  /// ownership.
  mlir::LogicalResult
  registerDynamicAttr(std::unique_ptr<DynamicAttributeImpl> attr);
  /// Lookup a DynamicAttribute with the given name. Returns nullptr if none
  /// is found.
  DynamicAttributeImpl *lookupAttr(llvm::StringRef name) const;

  /// Register a type alias.
  mlir::LogicalResult registerTypeAlias(TypeAlias typeAlias);
  /// Lookup a type alias. Returns nullptr if not found.
  TypeAlias *lookupTypeAlias(llvm::StringRef name) const;
  /// Register an attribute alias.
  mlir::LogicalResult registerAttrAlias(AttributeAlias attrAlias);
  /// Lookup an attribute alias. Returns nullptr if not found.
  AttributeAlias *lookupAttrAlias(llvm::StringRef name) const;

  /// Lookup dynamic type metadata associated with a type, if any.
  TypeMetadata *lookupTypeData(mlir::Type type);
  /// Lookup dynamic attribute metadata associated with an attribute, if any.
  AttributeMetadata *lookupAttributeData(mlir::Attribute attr);

  /// Query the objects of this dynamic dialect.
  std::vector<DynamicOperation *> getOps();
  std::vector<DynamicTypeImpl *> getTypes();
  std::vector<DynamicAttributeImpl *> getAttributes();

private:
  class Impl;
  std::unique_ptr<Impl> impl;

  friend class DynamicOperation;
};

} // end namespace dmc
