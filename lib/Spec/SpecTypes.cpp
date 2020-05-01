#include "dmc/Spec/SpecDialect.h"
#include "dmc/Spec/SpecTypes.h"
#include "dmc/Spec/Support.h"

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

/// Storage for SpecTypes parameterized by a width.
struct WidthStorage : public TypeStorage {
  /// Use width as key.
  using KeyTy = unsigned;

  explicit WidthStorage(KeyTy key) : width{key} {}

  /// Compare the width.
  bool operator==(const KeyTy &key) const { return key == width; }
  /// Hash the width.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  /// Create the WidthStorage;
  static WidthStorage *construct(TypeStorageAllocator &alloc,
                                 const KeyTy &key) {
    return new (alloc.allocate<WidthStorage>())
        WidthStorage{key};
  }

  KeyTy width;
};

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

} // end namespace detail

/// Helper functions.
namespace {

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

LogicalResult verifyTypeList(Location loc, ArrayRef<Type> tys) {
  if (tys.empty())
    return emitError(loc) << "empty Type list passed";
  llvm::SmallPtrSet<Type, 4> typeSet{std::begin(tys), std::end(tys)};
  if (std::size(typeSet) != std::size(tys))
    return emitError(loc) << "duplicate Types passed";
  return success();
}

} // end namespace impl

auto getSortedWidths(ArrayRef<unsigned> widths) {
  return getSortedListOf<std::less<unsigned>>(widths);
}

auto getSortedTypes(ArrayRef<Type> tys) {
  return getSortedListOf<detail::TypeComparator>(tys);
}

} // end anonymous namespace

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
AnyIType AnyIType::get(mlir::MLIRContext *ctx, unsigned width) {
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
AnyIntOfWidthsType AnyIntOfWidthsType::get(MLIRContext *ctx,
                                          ArrayRef<unsigned> widths) {
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
IType IType::get(MLIRContext *ctx, unsigned width) {
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
    MLIRContext *ctx, ArrayRef<unsigned> widths) {
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
SIType SIType::get(MLIRContext *ctx, unsigned width) {
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
    MLIRContext *ctx, ArrayRef<unsigned> widths) {
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
UIType UIType::get(MLIRContext *ctx, unsigned width) {
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
    MLIRContext *ctx, ArrayRef<unsigned> widths) {
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

FType FType::get(MLIRContext *ctx, unsigned width) {
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
    MLIRContext *ctx, ArrayRef<unsigned> widths) {
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
OpaqueType OpaqueType::get(MLIRContext *ctx, StringRef dialectName,
                           StringRef typeName) {
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

/// Type printing.
namespace {
void printSingleWidth(StringRef name, detail::WidthStorage *impl,
                      DialectAsmPrinter &printer) {
  printer << name << '<' << impl->width << '>';
}

void printWidthList(StringRef name, detail::WidthListStorage *impl, 
                    DialectAsmPrinter &printer) {
  auto it = std::begin(impl->widths);
  printer << name << '<' << (*it++);
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
    printSingleWidth("AnyI", type.cast<AnyIType>().getImpl(), printer);
    break;
  case AnyIntOfWidths:
    printWidthList("AnyIntOfWidths", type.cast<AnyIntOfWidthsType>().getImpl(), 
                   printer);
    break;
  case AnySignlessInteger:
    printer << "AnySignlessInteger";
    break;
  case I:
    printSingleWidth("I", type.cast<IType>().getImpl(), printer);
    break;
  case SignlessIntOfWidths:
    printWidthList("SignlessIntOfWidths", 
                   type.cast<SignlessIntOfWidthsType>().getImpl(), printer);
    break;
  case AnySignedInteger:
    printer << "AnySignedInteger";
    break;
  case SI:
    printSingleWidth("SI", type.cast<SIType>().getImpl(), printer);
    break;
  case SignedIntOfWidths:
    printWidthList("SignedIntOfWidths", 
                   type.cast<SignedIntOfWidthsType>().getImpl(), printer);
    break;
  case AnyUnsignedInteger:
    printer << "AnyUnsignedInteger";
    break;
  case UI:
    printSingleWidth("UI", type.cast<UIType>().getImpl(), printer);
    break;
  case UnsignedIntOfWidths:
    printWidthList("UnsignedIntOfWidths", 
                   type.cast<UnsignedIntOfWidthsType>().getImpl(), printer);
    break;
  case Index:
    printer << "Index";
    break;
  case AnyFloat:
    printer << "AnyFloat";
    break;
  case F:
    printSingleWidth("F", type.cast<FType>().getImpl(), printer);
    break;
  case FloatOfWidths:
    printWidthList("FloatOfWidths", type.cast<FloatOfWidthsType>().getImpl(),
                   printer);
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

} // end namespace dmc
