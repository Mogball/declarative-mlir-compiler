#include "dmc/Spec/SpecDialect.h"
#include "dmc/Spec/Support.h"
#include "dmc/Spec/SpecTypeSwitch.h"
#include "dmc/Spec/SpecTypeDetail.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Dynamic/DynamicType.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/IR/TypeUtilities.h>

using namespace mlir;

namespace dmc {

namespace detail {

/// Comparing types by their opaque pointers will ensure consitent
/// ordering throughout a single program lifetime.
struct TypeComparator {
  bool operator()(Type lhs, Type rhs) {
    return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
  }
};

/// Storage for SpecTypes parameterized by a list of Types. Lists must
/// be equal regardless of element order.
struct TypeListStorage : public TypeStorage {
  /// Compound key of all the contained types.
  using KeyTy = ImmutableSortedList<Type>;

  explicit TypeListStorage(KeyTy key) : types{std::move(key)} {}

  /// Compare all types.
  bool operator==(const KeyTy &key) const { return key == types; }
  /// Hash combined all types.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return key.hash();
  }

  /// Create the TypeListStorage.
  static TypeListStorage *construct(TypeStorageAllocator &alloc, KeyTy key) {
    return new (alloc.allocate<TypeListStorage>())
        TypeListStorage{std::move(key)};
  }

  KeyTy types;
};

/// WidthStorage implementation.
WidthStorage *WidthStorage::construct(TypeStorageAllocator &alloc,
                                      const KeyTy &key) {
  return new (alloc.allocate<WidthStorage>()) WidthStorage{key};
}

/// WidthListStorage implementation.
WidthListStorage *WidthListStorage::construct(TypeStorageAllocator &alloc,
                                              KeyTy key) {
  return new (alloc.allocate<WidthListStorage>())
      WidthListStorage{std::move(key)};
}

/// Storage for TypeConstraints with one Type parameter.
struct OneTypeStorage : public TypeStorage {
  /// Use the type as the key.
  using KeyTy = Type;

  explicit OneTypeStorage(KeyTy key) : type{key} {}

  /// Compare the Type.
  bool operator==(const KeyTy &key) const { return key == type; }
  /// Hash the type.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return hash_value(key);
  }

  /// Create the single Type storage.
  static OneTypeStorage *construct(TypeStorageAllocator &alloc,
                                   const KeyTy &key) {
    return new (alloc.allocate<OneTypeStorage>())
        OneTypeStorage{key};
  }

  KeyTy type;
};

/// Storage the Dialect and Type names.
struct OpaqueTypeStorage : public TypeStorage {
  /// Storage will only hold references.
  using KeyTy = std::pair<StringRef, StringRef>;

  explicit OpaqueTypeStorage(StringRef dialectName, StringRef typeName)
      : dialectName{dialectName}, typeName{typeName} {}

  bool operator==(const KeyTy &key) const {
    return key.first == dialectName && key.second == typeName;
  }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  static OpaqueTypeStorage *construct(TypeStorageAllocator &alloc,
      const KeyTy &key) {
    return new (alloc.allocate<OpaqueTypeStorage>())
        OpaqueTypeStorage{key.first, key.second};
  }

  StringRef dialectName;
  StringRef typeName;
};

/// Store a reference to a dialect and a belonging type.
struct IsaTypeStorage : public TypeStorage {
  using KeyTy = mlir::SymbolRefAttr;

  explicit IsaTypeStorage(mlir::SymbolRefAttr symRef)
      : symRef{symRef} {}

  bool operator==(const KeyTy &key) const { return key == symRef; }
  static llvm::hash_code hashKey(const KeyTy &key) { return hash_value(key); }

  static IsaTypeStorage *construct(TypeStorageAllocator &alloc,
                                   const KeyTy &key) {
    return new (alloc.allocate<IsaTypeStorage>()) IsaTypeStorage{key};
  }

  mlir::SymbolRefAttr symRef;

  StringRef getDialectRef() { return symRef.getRootReference(); }
  StringRef getTypeRef() { return symRef.getLeafReference(); }
};

} // end namespace detail

/// Helper functions.
namespace impl {

static LogicalResult verifyTypeList(Location loc, ArrayRef<Type> tys) {
  if (tys.empty())
    return emitError(loc) << "empty Type list passed";
  llvm::SmallPtrSet<Type, 4> typeSet{std::begin(tys), std::end(tys)};
  if (std::size(typeSet) != std::size(tys))
    return emitError(loc) << "duplicate Types passed";
  return success();
}

} // end namespace impl

static auto getSortedTypes(ArrayRef<Type> tys) {
  return getSortedListOf<detail::TypeComparator>(tys);
}

/// AnyOfType implementation.
AnyOfType AnyOfType::get(ArrayRef<Type> tys) {
  auto *ctx = tys.front().getContext();
  return Base::get(ctx, SpecTypes::AnyOf, getSortedTypes(tys));
}

AnyOfType AnyOfType::getChecked(Location loc, ArrayRef<Type> tys) {
  return Base::getChecked(loc, SpecTypes::AnyOf, getSortedTypes(tys));
}

LogicalResult AnyOfType::verifyConstructionInvariants(
    Location loc, ArrayRef<Type> tys) {
  return impl::verifyTypeList(loc, tys);
}

LogicalResult AnyOfType::verify(Type ty) {
  // Success if the Type is found
  for (auto baseTy : getImpl()->types) {
    if (SpecTypes::is(baseTy) &&
        succeeded(SpecTypes::delegateVerify(baseTy, ty)))
      return success();
    else if (baseTy == ty)
      return success();
  }
  return failure();
}

/// AllOfType implementation.
AllOfType AllOfType::get(ArrayRef<Type> tys) {
  auto *ctx = tys.front().getContext();
  return Base::get(ctx, SpecTypes::AllOf, getSortedTypes(tys));
}

AllOfType AllOfType::getChecked(Location loc, ArrayRef<Type> tys) {
  return Base::getChecked(loc, SpecTypes::AllOf, getSortedTypes(tys));
}

LogicalResult AllOfType::verifyConstructionInvariants(
    Location loc, ArrayRef<Type> tys) {
  return impl::verifyTypeList(loc, tys);
}

LogicalResult AllOfType::verify(Type ty) {
  for (auto baseTy : getImpl()->types) {
    if (SpecTypes::is(baseTy) &&
        failed(SpecTypes::delegateVerify(baseTy, ty)))
      return failure();
    else if (baseTy != ty)
      return failure();
  }
  return success();
}

/// AnyIType implementation.
LogicalResult AnyIType::verify(Type ty) {
  return success(ty.isInteger(getImpl()->width));
}

/// AnyIntOfWidthsType implementation.
LogicalResult AnyIntOfWidthsType::verify(Type ty) {
  return verifyWidthType(ty, [](unsigned width, Type ty) {
    return success(ty.isInteger(width));
  });
}

/// IType implementation.
LogicalResult IType::verify(Type ty) {
  return success(ty.isSignlessInteger(getImpl()->width));
}

/// SignlessIntOfWidthsType implementation.
LogicalResult SignlessIntOfWidthsType::verify(Type ty) {
  return verifyWidthType(ty, [](unsigned width, Type ty) {
    return success(ty.isSignlessInteger(width));
  });
}

/// SIType implementation.
LogicalResult SIType::verify(Type ty) {
  return success(ty.isSignedInteger(getImpl()->width));
}

/// SignedIntOfWidthsType implementation.
LogicalResult SignedIntOfWidthsType::verify(Type ty) {
  return verifyWidthType(ty, [](unsigned width, Type ty) {
    return success(ty.isSignedInteger(width));
  });
}

/// UIType implementation.
LogicalResult UIType::verify(Type ty) {
  return success(ty.isUnsignedInteger(getImpl()->width));
}

/// UnsignedIntOfWidthsType implementation.
LogicalResult UnsignedIntOfWidthsType::verify(Type ty) {
  return verifyWidthType(ty, [](unsigned width, Type ty) {
    return success(ty.isUnsignedInteger(width));
  });
}

/// FType implementation.
LogicalResult FType::verifyConstructionInvariants(
    Location loc, unsigned width) {
  return impl::verifyFloatWidth(loc, width);
}

LogicalResult FType::verify(Type ty) {
  return impl::verifyFloatType(getImpl()->width, ty);
}

/// FLoatOfWidthsType implementation
LogicalResult FloatOfWidthsType::verifyConstructionInvariants(
    Location loc, ArrayRef<unsigned> widths) {
  return impl::verifyWidthList(loc, widths, impl::verifyFloatWidth);
}

LogicalResult FloatOfWidthsType::verify(Type ty) {
  return verifyWidthType(ty, [](unsigned width, Type ty) {
    return impl::verifyFloatType(width, ty);
  });
}

/// ComplexType implementation.
ComplexType ComplexType::get(Type elTy) {
  return Base::get(elTy.getContext(), SpecTypes::Complex, elTy);
}

ComplexType ComplexType::getChecked(Location loc, Type elTy) {
  return Base::getChecked(loc, SpecTypes::Complex, elTy);
}

LogicalResult ComplexType::verify(Type ty) {
  // Check that the Type is a ComplexType
  if (auto complexTy = ty.dyn_cast<mlir::ComplexType>()) {
    auto elTyBase = getImpl()->type;
    auto elTy = complexTy.getElementType();
    if (SpecTypes::is(elTyBase))
      return SpecTypes::delegateVerify(elTyBase, elTy);
    else
      return success(elTyBase == elTy);
  }
  return failure();
}

/// OpaqueType implementation.
OpaqueType OpaqueType::get(StringRef dialectName, StringRef typeName,
                           MLIRContext *ctx) {
  return Base::get(ctx, SpecTypes::Opaque, dialectName, typeName);
}

OpaqueType OpaqueType::getChecked(Location loc, StringRef dialectName,
                                  StringRef typeName) {
  return Base::getChecked(loc, SpecTypes::Opaque, dialectName, typeName);
}

LogicalResult OpaqueType::verifyConstructionInvariants(
    Location loc, StringRef dialectName, StringRef typeName) {
  if (dialectName.empty())
    return emitError(loc) << "dialect name cannot be empty";
  if (typeName.empty())
    return emitError(loc) << "type name cannot be empty";
  return success();
}

LogicalResult OpaqueType::verify(Type ty) {
  return success(mlir::isOpaqueTypeWithName(
        ty, getImpl()->dialectName, getImpl()->typeName));
}

/// VariadicType implementation.
VariadicType VariadicType::get(Type ty) {
  return Base::get(ty.getContext(), SpecTypes::Variadic, ty);
}

VariadicType VariadicType::getChecked(Location loc, Type ty) {
  return Base::getChecked(loc, SpecTypes::Variadic, ty);
}

LogicalResult VariadicType::verifyConstructionInvariants(
    Location loc, Type ty) {
  /// TODO Need to assert that Variadic is used only as a top-level Type
  /// constraint, since it is more of a marker than a constraint. This is how
  /// TableGen does it. Nesting Variadic is illegal but a no-op anyway.
  ///
  /// Might be worth looking into argument attributes:
  ///
  /// dmc.Op @MyOp(%0 : !dmc.AnyInteger {variadic = true})
  ///
  if (!ty)
    return emitError(loc) << "type cannot be null";
  return success();
}

LogicalResult VariadicType::verify(Type ty) {
  auto baseTy = getImpl()->type;
  if (SpecTypes::is(baseTy))
    return SpecTypes::delegateVerify(baseTy, ty);
  else if (baseTy != ty)
    return failure();
  return success();
}

/// IsaType implementation.
IsaType IsaType::get(mlir::SymbolRefAttr typeRef) {
  return Base::get(typeRef.getContext(), SpecTypes::Isa, typeRef);
}

IsaType IsaType::getChecked(Location loc, mlir::SymbolRefAttr typeRef) {
  return Base::getChecked(loc, SpecTypes::Isa, typeRef);
}

LogicalResult IsaType::verifyConstructionInvariants(
    Location loc, mlir::SymbolRefAttr typeRef) {
  if (!typeRef)
    return emitError(loc) << "Null type reference";
  auto nestedRefs = typeRef.getNestedReferences();
  if (llvm::size(nestedRefs) != 1)
    return emitError(loc) << "Expected type reference to have depth 2, "
        << "in the form @dialect::@typename";
  /// TODO verify that the type reference is valid. Dynamic dialects and
  /// types are not registered during first-parse so currently this must
  /// be deferred to IsaType::verify.
  return success();
}

static DynamicTypeImpl *lookupTypeReference(
    MLIRContext *ctx, StringRef dialectName, StringRef typeName) {
  /// First resolve the dialect reference.
  /// TODO implement a post-parse verification pass?
  auto *dialect = ctx->getRegisteredDialect(dialectName);
  if (!dialect) {
    llvm::errs() << "error: reference to unknown dialect '"
        << dialectName << '\'';
    return nullptr;
  }
  auto *dynDialect = dynamic_cast<DynamicDialect *>(dialect);
  if (!dynDialect) {
    llvm::errs() << "error: dialect '" << dialectName
        << "' is not a dynamic dialect";
    return nullptr;
  }
  /// Resolve the type reference.
  auto *typeImpl = dynDialect->lookupType(typeName);
  if (!typeImpl) {
    llvm::errs() << "error: dialect '" << dialectName
        << "' does not have type '" << typeName << '\'';
    return nullptr;
  }
  return typeImpl;
}

LogicalResult IsaType::verify(Type ty) {
  /// Check that the argument type is a dynamic type.
  auto dynTy = ty.dyn_cast<DynamicType>();
  if (!dynTy)
    return failure();
  /// Lookup the type kind.
  auto *typeImpl = lookupTypeReference(
      getContext(), getImpl()->getDialectRef(), getImpl()->getTypeRef());
  if (!typeImpl)
    return failure();
  /// Simply compare the type kinds.
  return success(typeImpl == dynTy.getTypeImpl());
}

/// Type printing.
struct PrintAction {
  DialectAsmPrinter &printer;

  template <typename ConcreteType>
  int operator()(ConcreteType base) const {
    base.print(printer);
    return 0;
  }
};

void SpecDialect::printType(Type type, DialectAsmPrinter &printer) const {
  PrintAction action{printer};
  SpecTypes::kindSwitch(action, type);
}

void printTypeList(ArrayRef<Type> tys, DialectAsmPrinter &printer) {
  auto it = std::begin(tys);
  printer << '<';
  printer.printType(*it++);
  for (auto e = std::end(tys); it != e; ++it) {
    printer << ',';
    printer.printType(*it);
  }
  printer << '>';
}

void AnyOfType::print(DialectAsmPrinter &printer) {
  printer << getTypeName();
  printTypeList(getImpl()->types, printer);
}

void AllOfType::print(DialectAsmPrinter &printer) {
  printer << getTypeName();
  printTypeList(getImpl()->types, printer);
}

void ComplexType::print(DialectAsmPrinter &printer) {
  printer << getTypeName() << '<';
  printer.printType(getImpl()->type);
  printer << '>';
}

void OpaqueType::print(DialectAsmPrinter &printer) {
  printer << getTypeName() << "<\"" << getImpl()->dialectName << "\",\""
          << getImpl()->typeName << "\">";
}

void VariadicType::print(DialectAsmPrinter &printer) {
  printer << getTypeName() << '<';
  printer.printType(getImpl()->type);
  printer << '>';
}

void IsaType::print(DialectAsmPrinter &printer) {
  printer << getTypeName() << '<';
  printer.printAttribute(getImpl()->symRef);
  printer << '>';
}

} // end namespace dmc
