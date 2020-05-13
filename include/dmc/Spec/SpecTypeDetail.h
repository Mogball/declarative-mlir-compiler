#pragma once

#include "SpecTypeImplementation.h"
#include "Support.h"

namespace dmc {

namespace detail {

/// Storage for SpecTypes parameterized by a width.
struct WidthStorage : public mlir::TypeStorage {
  /// Use width as key.
  using KeyTy = unsigned;

  explicit inline WidthStorage(KeyTy key) : width{key} {}

  /// Compare widths.
  inline bool operator==(const KeyTy &key) const { return key == width; }
  /// Hash the width.
  static llvm::hash_code hashKey(const KeyTy &key)
  { return llvm::hash_value(key); }
  /// Create the WidthStorage;
  static WidthStorage *construct(mlir::TypeStorageAllocator &alloc,
                                 const KeyTy &key);

  KeyTy width;
};

/// Storage for SpecTypes parameterized by a list of widths. Used
/// commonly for Integer TypeConstraints. Lists must be equal
/// regardless of element order.
struct WidthListStorage : public mlir::TypeStorage {
  /// Use list of widths as a compound key.
  using KeyTy = ImmutableSortedList<unsigned>;

  explicit inline WidthListStorage(KeyTy key) : widths{std::move(key)} {}

  /// Compare all widths.
  inline bool operator==(const KeyTy &key) const { return key == widths; }
  /// Hash all the widths together.
  static llvm::hash_code hashKey(const KeyTy &key) { return key.hash(); }
  /// Create the WidthListStorage.
  static WidthListStorage *construct(mlir::TypeStorageAllocator &alloc,
                                     KeyTy key);

  KeyTy widths;
};

} // end namespace detail

namespace impl {
mlir::LogicalResult verifyIntWidth(mlir::Location loc, unsigned width);
mlir::LogicalResult verifyFloatWidth(mlir::Location loc, unsigned width);
mlir::LogicalResult verifyFloatType(unsigned width, mlir::Type ty);
mlir::LogicalResult verifyWidthList(
    mlir::Location loc, llvm::ArrayRef<unsigned> widths,
    mlir::LogicalResult (&verifyWidth)(mlir::Location, unsigned));

llvm::Optional<unsigned> parseSingleWidth(mlir::DialectAsmParser &parser);
void printSingleWidth(mlir::DialectAsmPrinter &printer, unsigned width);

using WidthList = llvm::SmallVector<unsigned, 2>;
llvm::Optional<WidthList> parseWidthList(
    mlir::DialectAsmParser &parser);
void printWidthList(mlir::DialectAsmPrinter &printer,
                    llvm::ArrayRef<unsigned> widths);
} // end namespace impl

/// A numeric type with a fixed bit-width.
template <typename ConcreteType, unsigned Kind>
class NumericWidthType : public SpecType<ConcreteType, Kind,
                                         detail::WidthStorage> {
public:
  using Base = NumericWidthType<ConcreteType, Kind>;
  using Parent = SpecType<ConcreteType, Kind, detail::WidthStorage>;
  using Parent::Parent;

  static ConcreteType get(unsigned width, mlir::MLIRContext *ctx) {
    return Parent::get(ctx, Kind, width);
  }

  static ConcreteType getChecked(mlir::Location loc, unsigned width) {
    return Parent::getChecked(loc, Kind, width);
  }

  static mlir::Type parse(mlir::DialectAsmParser &parser) {
    auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
    auto width = impl::parseSingleWidth(parser);
    if (!width)
      return {};
    return getChecked(loc, *width);
  }

  void print(mlir::DialectAsmPrinter &printer) {
    printer << ConcreteType::getTypeName();
    impl::printSingleWidth(printer, this->getImpl()->width);
  }
};

/// An integer type with fixed width.
template <typename ConcreteType, unsigned Kind>
class IntegerWidthType : public NumericWidthType<ConcreteType, Kind> {
public:
  using Base = IntegerWidthType<ConcreteType, Kind>;
  using Parent = NumericWidthType<ConcreteType, Kind>;
  using Parent::Parent;

  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, unsigned width) {
    return impl::verifyIntWidth(loc, width);
  }
};

/// A numeric type of one of a list of specified widths.
template <typename ConcreteType, unsigned Kind>
class NumericTypeOfWidths : public SpecType<ConcreteType, Kind,
                                            detail::WidthListStorage> {
public:
  using Base = NumericTypeOfWidths<ConcreteType, Kind>;
  using Parent = SpecType<ConcreteType, Kind, detail::WidthListStorage>;
  using Parent::Parent;

  static ConcreteType get(llvm::ArrayRef<unsigned> widths,
                          mlir::MLIRContext *ctx) {
    return Parent::get(ctx, Kind, getSortedWidths(widths));
  }

  static ConcreteType getChecked(mlir::Location loc,
                                 llvm::ArrayRef<unsigned> widths) {
    return Parent::getChecked(loc, Kind, getSortedWidths(widths));
  }

  static mlir::Type parse(mlir::DialectAsmParser &parser) {
    auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
    auto widths = impl::parseWidthList(parser);
    if (!widths)
      return {};
    return getChecked(loc, *widths);
  }

  void print(mlir::DialectAsmPrinter &printer) {
    printer << ConcreteType::getTypeName();
    impl::printWidthList(printer, this->getImpl()->widths);
  }

protected:
  mlir::LogicalResult verifyWidthType(
      mlir::Type ty, mlir::LogicalResult (verifyType)(unsigned, mlir::Type)) {
    for (auto width : this->getImpl()->widths)
      if (mlir::succeeded(verifyType(width, ty)))
        return mlir::success();
    return mlir::failure();
  }

private:
  static auto getSortedWidths(llvm::ArrayRef<unsigned> widths) {
    return getSortedListOf<std::less<unsigned>>(widths);
  }
};

/// An integer type of one of one a list of specified widths.
template <typename ConcreteType, unsigned Kind>
class IntegerTypeOfWidths : public NumericTypeOfWidths<ConcreteType, Kind> {
public:
  using Base = IntegerTypeOfWidths<ConcreteType, Kind>;
  using Parent = NumericTypeOfWidths<ConcreteType, Kind>;
  using Parent::Parent;

  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, llvm::ArrayRef<unsigned> widths) {
    return impl::verifyWidthList(loc, widths, impl::verifyIntWidth);
  }
};

} // end namespace dmc
