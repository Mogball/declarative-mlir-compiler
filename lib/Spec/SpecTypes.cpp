#include "dmc/Spec/SpecTypes.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/TypeUtilities.h>

#include <unordered_set>

using namespace mlir;

namespace dmc {

namespace detail {

/// An immutable list that self-sorts on creation.
template <typename T>
struct ImmutableSortedList : public llvm::SmallVector<T, 4> {
  /// Sort on creation with comparator.
  template <typename Container, typename ComparatorT>
  ImmutableSortedList(const Container &c,
                      ComparatorT comparator = ComparatorT{})
      : llvm::SmallVector<T, 4>{std::begin(c), std::end(c)} {
    llvm::sort(this->begin(), this->end(), comparator);
  }

  /// Compare list sizes and contents.
  bool operator==(const ImmutableSortedList<T> &other) const {
    if (this->size() != other.size())
      return false;
    return std::equal(this->begin(), this->end(), other.begin());
  }

  /// Hash list values.
  llvm::hash_code hash() const {
    return llvm::hash_combine_range(this->begin(), this->end());
  }
};

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
  static TypeListStorage *construct(TypeStorageAllocator &alloc,
                                    const KeyTy &key) {
    return new (alloc.allocate<TypeListStorage>())
        TypeListStorage{key};
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
  static WidthListStorage *construct(TypeStorageAllocator &alloc,
                                     const KeyTy &key) {
    return new (alloc.allocate<WidthListStorage>())
        WidthListStorage{key};
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
    return hash_value(Type{key});
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

auto getSortedWidths(ArrayRef<unsigned> widths) {
  return detail::ImmutableSortedList<unsigned>{widths, std::less<unsigned>{}};
}

} // end anonymous namespace

/// AnyOfType implementation.
AnyOfType AnyOfType::get(ArrayRef<Type> tys) {
  auto *ctx = tys.front().getContext();
  detail::ImmutableSortedList<Type> types{tys, detail::TypeComparator{}};
  return Base::get(ctx, SpecTypes::AnyOf, std::move(types));
}

AnyOfType AnyOfType::getChecked(Location loc, ArrayRef<Type> tys) {
  detail::ImmutableSortedList<Type> types{tys, detail::TypeComparator{}};
  return Base::getChecked(loc, SpecTypes::AnyOf, std::move(types));
}

LogicalResult AnyOfType::verifyConstructionInvariants(
    Location loc, ArrayRef<Type> tys) {
  if (tys.empty())
    return emitError(loc) << "empty Type list passed to 'AnyOf'";
  llvm::SmallPtrSet<Type, 4> types{std::begin(tys), std::end(tys)};
  if (std::size(types) != std::size(tys))
    return emitError(loc) << "duplicate Types in list passed to 'AnyOf'";
  return success();
}

LogicalResult AnyOfType::verify(Type ty) {
  // Success if the Type is found
  auto &types = getImpl()->types;
  return success(llvm::find(types, ty) != std::end(types));
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
  return verifyWidthList<AnyIType>(loc, widths, "integer");
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
  return verifyWidthList<FType>(loc, widths, "float");
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

} // end namespace dmc
