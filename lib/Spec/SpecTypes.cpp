#include "dmc/Spec/SpecDialect.h"
#include "dmc/Spec/SpecTypes.h"
#include "dmc/Spec/Support.h"
#include "dmc/Spec/SpecTypeDetail.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Dynamic/DynamicType.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/TypeUtilities.h>

#include <unordered_set>

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
WidthStorage::WidthStorage(KeyTy key) : width{key} {}

/// Compare the width.
bool WidthStorage::operator==(const KeyTy &key) const {
  return key == width;
}

/// Hash the width.
llvm::hash_code WidthStorage::hashKey(const KeyTy &key) {
  return llvm::hash_value(key);
}

/// Create the WidthStorage;
WidthStorage *WidthStorage::construct(
    TypeStorageAllocator &alloc, const KeyTy &key) {
  return new (alloc.allocate<WidthStorage>())
      WidthStorage{key};
}

/// Storage for SpecTypes parameterized by a list of widths. Used
/// commonly for Integer TypeConstraints. Lists must be equal
/// regardless of element order.
struct WidthListStorage : public TypeStorage {
  /// Use list of widths as a compound key.
  using KeyTy = ImmutableSortedList<unsigned>;

  explicit WidthListStorage(KeyTy key) : widths{std::move(key)} {}

  /// Compare all types.
  bool operator==(const KeyTy &key) const { return key == widths; }
  /// Hash all the widths together.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return key.hash();
  }

  /// Create the WidthListStorage.
  static WidthListStorage *construct(TypeStorageAllocator &alloc, KeyTy key) {
    return new (alloc.allocate<WidthListStorage>())
        WidthListStorage{std::move(key)};
  }

  KeyTy widths;
};

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

  bool operator==(const KeyTy &key) const {
    return key == symRef;
  }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return hash_value(key);
  }

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

template <typename BaseT>
LogicalResult verifyWidthList(Location loc, ArrayRef<unsigned> widths,
                              const char *typeName) {
  if (widths.empty())
    return emitError(loc) << "empty " << typeName << " width list passed";
  std::unordered_set<unsigned> widthSet{std::begin(widths),
                                        std::end(widths)};
  if (std::size(widthSet) != std::size(widths))
    return emitError(loc) << "duplicate " << typeName << " widths passed";

  for (auto width : widths) {
    if (failed(BaseT::verifyConstructionInvariants(loc, width)))
      return failure();
  }
  return success();
}

static LogicalResult verifyTypeList(Location loc, ArrayRef<Type> tys) {
  if (tys.empty())
    return emitError(loc) << "empty Type list passed";
  llvm::SmallPtrSet<Type, 4> typeSet{std::begin(tys), std::end(tys)};
  if (std::size(typeSet) != std::size(tys))
    return emitError(loc) << "duplicate Types passed";
  return success();
}

} // end namespace impl

static auto getSortedWidths(ArrayRef<unsigned> widths) {
  return getSortedListOf<std::less<unsigned>>(widths);
}

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
AnyIType AnyIType::get(unsigned width, MLIRContext *ctx) {
  return Base::get(ctx, SpecTypes::AnyI, width);
}

AnyIType AnyIType::getChecked(Location loc, unsigned width) {
  return Base::getChecked(loc, SpecTypes::AnyI, width);
}

LogicalResult AnyIType::verifyConstructionInvariants(
    Location loc, unsigned width) {
  switch (width) {
  case 1:
  case 8:
  case 16:
  case 32:
  case 64:
    return success();
  default:
    return emitError(loc) << "width must be one of [1, 8, 16, 32, 64]";
  }
}

LogicalResult AnyIType::verify(Type ty) {
  return success(ty.isInteger(getImpl()->width));
}

/// AnyIntOfWidthsType implementation.
AnyIntOfWidthsType AnyIntOfWidthsType::get(
    ArrayRef<unsigned> widths, MLIRContext *ctx) {
  return Base::get(ctx, SpecTypes::AnyIntOfWidths,
                   getSortedWidths(widths));
}

AnyIntOfWidthsType AnyIntOfWidthsType::getChecked(Location loc,
                                           ArrayRef<unsigned> widths) {
  return Base::getChecked(loc, SpecTypes::AnyIntOfWidths,
                          getSortedWidths(widths));

}

LogicalResult AnyIntOfWidthsType::verifyConstructionInvariants(
    Location loc, ArrayRef<unsigned> widths) {
  return impl::verifyWidthList<AnyIType>(loc, widths, "integer");
}

LogicalResult AnyIntOfWidthsType::verify(Type ty) {
  for (auto width : getImpl()->widths) {
    if (ty.isInteger(width))
      return success();
  }
  return failure();
}

/// IType implementation.
IType IType::get(unsigned width, MLIRContext *ctx) {
  return Base::get(ctx, SpecTypes::I, width);
}

IType IType::getChecked(Location loc, unsigned width) {
  return Base::getChecked(loc, SpecTypes::I, width);
}

LogicalResult IType::verifyConstructionInvariants(
    Location loc, unsigned width) {
  return AnyIType::verifyConstructionInvariants(loc, width);
}

LogicalResult IType::verify(Type ty) {
  return success(ty.isSignlessInteger(getImpl()->width));
}

/// SignlessIntOfWidthsType implementation.
SignlessIntOfWidthsType SignlessIntOfWidthsType::get(
    ArrayRef<unsigned> widths, MLIRContext *ctx) {
  return Base::get(ctx, SpecTypes::SignlessIntOfWidths,
                   getSortedWidths(widths));
}

SignlessIntOfWidthsType SignlessIntOfWidthsType::getChecked(
    Location loc, ArrayRef<unsigned> widths) {
  return Base::getChecked(loc, SpecTypes::SignlessIntOfWidths,
                          getSortedWidths(widths));
}

LogicalResult SignlessIntOfWidthsType::verifyConstructionInvariants(
    Location loc, ArrayRef<unsigned> widths) {
  return AnyIntOfWidthsType::verifyConstructionInvariants(loc, widths);
}

LogicalResult SignlessIntOfWidthsType::verify(Type ty) {
  for (auto width : getImpl()->widths) {
    if (ty.isSignlessInteger(width))
      return success();
  }
  return failure();
}

/// SIType implementation.
SIType SIType::get(unsigned width, MLIRContext *ctx) {
  return Base::get(ctx, SpecTypes::SI, width);
}

SIType SIType::getChecked(Location loc, unsigned width) {
  return Base::getChecked(loc, SpecTypes::SI, width);
}

LogicalResult SIType::verifyConstructionInvariants(
    Location loc, unsigned width) {
  return AnyIType::verifyConstructionInvariants(loc, width);
}

LogicalResult SIType::verify(Type ty) {
  return success(ty.isSignedInteger(getImpl()->width));
}

/// SignedIntOfWidthsType implementation.
SignedIntOfWidthsType SignedIntOfWidthsType::get(
    ArrayRef<unsigned> widths, MLIRContext *ctx) {
  return Base::get(ctx, SpecTypes::SignedIntOfWidths,
                   getSortedWidths(widths));
}

SignedIntOfWidthsType SignedIntOfWidthsType::getChecked(
    Location loc, ArrayRef<unsigned> widths) {
  return Base::getChecked(loc, SpecTypes::SignedIntOfWidths,
                          getSortedWidths(widths));
}

LogicalResult SignedIntOfWidthsType::verifyConstructionInvariants(
    Location loc, ArrayRef<unsigned> widths) {
  return AnyIntOfWidthsType::verifyConstructionInvariants(loc, widths);
}

LogicalResult SignedIntOfWidthsType::verify(Type ty) {
  for (auto width : getImpl()->widths) {
    if (ty.isSignedInteger(width))
      return success();
  }
  return failure();
}

/// UIType implementation.
UIType UIType::get(unsigned width, MLIRContext *ctx) {
  return Base::get(ctx, SpecTypes::UI, width);
}

UIType UIType::getChecked(Location loc, unsigned width) {
  return Base::getChecked(loc, SpecTypes::UI, width);
}

LogicalResult UIType::verifyConstructionInvariants(
    Location loc, unsigned width) {
  return AnyIType::verifyConstructionInvariants(loc, width);
}

LogicalResult UIType::verify(Type ty) {
  return success(ty.isUnsignedInteger(getImpl()->width));
}

/// UnsignedIntOfWidthsType implementation.
UnsignedIntOfWidthsType UnsignedIntOfWidthsType::get(
    ArrayRef<unsigned> widths, MLIRContext *ctx) {
  return Base::get(ctx, SpecTypes::UnsignedIntOfWidths,
                   getSortedWidths(widths));
}

UnsignedIntOfWidthsType UnsignedIntOfWidthsType::getChecked(
    Location loc, ArrayRef<unsigned> widths) {
  return Base::getChecked(loc, SpecTypes::UnsignedIntOfWidths,
                          getSortedWidths(widths));
}

LogicalResult UnsignedIntOfWidthsType::verifyConstructionInvariants(
    Location loc, ArrayRef<unsigned> widths) {
  return AnyIntOfWidthsType::verifyConstructionInvariants(loc, widths);
}

LogicalResult UnsignedIntOfWidthsType::verify(Type ty) {
  for (auto width : getImpl()->widths) {
    if (ty.isUnsignedInteger(width))
      return success();
  }
  return failure();
}

/// FType implementation.
namespace {
inline LogicalResult verifyFloatWidth(unsigned width, Type ty) {
  switch (width) {
  case 16:
    return success(ty.isF16());
  case 32:
    return success(ty.isF32());
  case 64:
    return success(ty.isF64());
  default:
    llvm_unreachable("Invalid floating point width");
    return failure();
  }
}
} // end anonymous namespace

FType FType::get(unsigned width, MLIRContext *ctx) {
  return Base::get(ctx, SpecTypes::F, width);
}

FType FType::getChecked(Location loc, unsigned width) {
  return Base::getChecked(loc, SpecTypes::F, width);
}

LogicalResult FType::verifyConstructionInvariants(
    Location loc, unsigned width) {
  // Width must be one of [16, 32, 64]
  switch (width) {
  case 16:
  case 32:
  case 64:
    return success();
  default:
    return emitError(loc) << "float width must be one of [16, 32, 64]";
  }
}

LogicalResult FType::verify(Type ty) {
  return verifyFloatWidth(getImpl()->width, ty);
}

/// FLoatOfWidthsType implementation
FloatOfWidthsType FloatOfWidthsType::get(
    ArrayRef<unsigned> widths, MLIRContext *ctx) {
  return Base::get(ctx, SpecTypes::FloatOfWidths,
                   getSortedWidths(widths));
}

FloatOfWidthsType FloatOfWidthsType::getChecked(
    Location loc, ArrayRef<unsigned> widths) {
  return Base::getChecked(loc, SpecTypes::FloatOfWidths,
                          getSortedWidths(widths));
}

LogicalResult FloatOfWidthsType::verifyConstructionInvariants(
    Location loc, ArrayRef<unsigned> widths) {
  return impl::verifyWidthList<FType>(loc, widths, "float");
}

LogicalResult FloatOfWidthsType::verify(Type ty) {
  for (auto width : getImpl()->widths) {
    if (succeeded(verifyFloatWidth(width, ty)))
      return success();
  }
  return failure();
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
void SpecDialect::printSingleWidth(detail::WidthStorage *impl,
                                   DialectAsmPrinter &printer) const {
  printer << '<' << impl->width << '>';
}

namespace {
void printWidthList(detail::WidthListStorage *impl,
                    DialectAsmPrinter &printer) {
  auto it = std::begin(impl->widths);
  printer << '<' << (*it++);
  for (auto e = std::end(impl->widths); it != e; ++it)
    printer << ',' << (*it);
  printer << '>';
}
} // end anonymous namespace

void SpecDialect::printType(Type type, DialectAsmPrinter &printer) const {
  using namespace SpecTypes;
  assert(is(type) && "Not a SpecType");
  switch (type.getKind()) {
  case Any:
    printer << "Any";
    break;
  case None:
    printer << "None";
    break;
  case AnyOf:
    type.cast<AnyOfType>().print(printer);
    break;
  case AllOf:
    type.cast<AllOfType>().print(printer);
  case AnyInteger:
    printer << "AnyInteger";
    break;
  case AnyI:
    printer << "AnyI";
    printSingleWidth(type.cast<AnyIType>().getImpl(), printer);
    break;
  case AnyIntOfWidths:
    printer << "AnyIntOfWidths";
    printWidthList(type.cast<AnyIntOfWidthsType>().getImpl(), printer);
    break;
  case AnySignlessInteger:
    printer << "AnySignlessInteger";
    break;
  case I:
    printer << "I";
    printSingleWidth(type.cast<IType>().getImpl(), printer);
    break;
  case SignlessIntOfWidths:
    printer << "SignlessIntOfWidths";
    printWidthList(type.cast<SignlessIntOfWidthsType>().getImpl(), printer);
    break;
  case AnySignedInteger:
    printer << "AnySignedInteger";
    break;
  case SI:
    printer << "SI";
    printSingleWidth(type.cast<SIType>().getImpl(), printer);
    break;
  case SignedIntOfWidths:
    printer << "SignedIntOfWidths";
    printWidthList(type.cast<SignedIntOfWidthsType>().getImpl(), printer);
    break;
  case AnyUnsignedInteger:
    printer << "AnyUnsignedInteger";
    break;
  case UI:
    printer << "UI";
    printSingleWidth(type.cast<UIType>().getImpl(), printer);
    break;
  case UnsignedIntOfWidths:
    printer << "UnsignedIntOfWidths";
    printWidthList(type.cast<UnsignedIntOfWidthsType>().getImpl(), printer);
    break;
  case Index:
    printer << "Index";
    break;
  case AnyFloat:
    printer << "AnyFloat";
    break;
  case F:
    printer << "F";
    printSingleWidth(type.cast<FType>().getImpl(), printer);
    break;
  case FloatOfWidths:
    printer << "FloatOfWidths";
    printWidthList(type.cast<FloatOfWidthsType>().getImpl(), printer);
    break;
  case BF16:
    printer << "BF16";
    break;
  case AnyComplex:
    printer << "AnyComplex";
    break;
  case Complex:
    type.cast<ComplexType>().print(printer);
    break;
  case Opaque:
    type.cast<OpaqueType>().print(printer);
    break;
  case Variadic:
    type.cast<VariadicType>().print(printer);
    break;
  case Isa:
    type.cast<IsaType>().print(printer);
    break;
  default:
    llvm_unreachable("Unknown SpecType");
    break;
  }
}

void printTypeList(ArrayRef<Type> tys, DialectAsmPrinter &printer) {
  auto it = std::begin(tys);
  printer.printType(*it++);
  for (auto e = std::end(tys); it != e; ++it) {
    printer << ',';
    printer.printType(*it);
  }
}

void AnyOfType::print(DialectAsmPrinter &printer) {
  printer << "AnyOf<";
  printTypeList(getImpl()->types, printer);
  printer << '>';
}

void AllOfType::print(DialectAsmPrinter &printer) {
  printer << "AllOf<";
  printTypeList(getImpl()->types, printer);
  printer << '>';
}

void ComplexType::print(DialectAsmPrinter &printer) {
  printer << "Complex<";
  printer.printType(getImpl()->type);
  printer << '>';
}

void OpaqueType::print(DialectAsmPrinter &printer) {
  printer << "Opaque<" << getImpl()->dialectName << ','
          << getImpl()->typeName << '>';
}

void VariadicType::print(DialectAsmPrinter &printer) {
  printer << "Variadic<";
  printer.printType(getImpl()->type);
  printer << '>';
}

void IsaType::print(DialectAsmPrinter &printer) {
  printer << "Isa<";
  printer.printAttribute(getImpl()->symRef);
  printer << '>';
}

} // end namespace dmc
