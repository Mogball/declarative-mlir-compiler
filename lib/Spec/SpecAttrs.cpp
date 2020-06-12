#include "dmc/Spec/SpecAttrSwitch.h"
#include "dmc/Spec/SpecDialect.h"
#include "dmc/Spec/SpecTypeDetail.h"
#include "dmc/Spec/SpecTypeImplementation.h"
#include "dmc/Dynamic/DynamicAttribute.h"
#include "dmc/Dynamic/DynamicDialect.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;

namespace dmc {

namespace detail {

/// TypedAttrStorage implementation. Store a type or type constraint.
TypedAttrStorage::TypedAttrStorage(KeyTy key) : type{key} {}

bool TypedAttrStorage::operator==(const KeyTy &key) const {
  return key == type;
}

llvm::hash_code TypedAttrStorage::hashKey(const KeyTy &key) {
  return hash_value(key);
}

TypedAttrStorage *TypedAttrStorage::construct(AttributeStorageAllocator &alloc,
                                              const KeyTy &key) {
  return new (alloc.allocate<TypedAttrStorage>())
      TypedAttrStorage{key};
}

/// AttrListStorage implementation. Store a list of attributes.
struct AttrListStorage : public AttributeStorage {
  using KeyTy = ImmutableSortedList<Attribute>;

  explicit AttrListStorage(KeyTy key) : attrs{std::move(key)} {}
  bool operator==(const KeyTy &key) const { return key == attrs; }
  static llvm::hash_code hashKey(const KeyTy &key) { return key.hash(); }

  static AttrListStorage *construct(AttributeStorageAllocator &alloc,
                                    KeyTy key) {
    return new (alloc.allocate<AttrListStorage>())
      AttrListStorage{std::move(key)};
  }

  KeyTy attrs;
};

/// OneTypeAttrStorage implementation. Store one type.
struct OneTypeAttrStorage : public AttributeStorage {
  using KeyTy = Type;

  explicit OneTypeAttrStorage(KeyTy key) : type{key} {}
  bool operator==(const KeyTy &key) const { return key == type; }
  static llvm::hash_code hashKey(const KeyTy &key) { return hash_value(key); }

  static OneTypeAttrStorage *construct(AttributeStorageAllocator &alloc,
                                       KeyTy key) {
    return new (alloc.allocate<OneTypeAttrStorage>())
        OneTypeAttrStorage{key};
  }

  KeyTy type;
};

struct DimensionAttrStorage : public AttributeStorage {
  using KeyTy = ArrayRef<int64_t>;

  explicit DimensionAttrStorage(KeyTy key) : dims{key} {}
  static llvm::hash_code hashKey(KeyTy key) { return hash_value(key); }

  bool operator==(KeyTy key) const { return key == dims; }

  static DimensionAttrStorage *construct(AttributeStorageAllocator &alloc,
                                         KeyTy key) {
    auto dims = alloc.copyInto(key);
    return new (alloc.allocate<DimensionAttrStorage>())
        DimensionAttrStorage{dims};
  }

  KeyTy dims;
};

/// DefaultAttrStorage implementation. Store the base attribute or constraint
/// and the default value.
struct DefaultAttrStorage : public AttributeStorage {
  using KeyTy = std::pair<Attribute, Attribute>;

  explicit DefaultAttrStorage(Attribute baseAttr, Attribute defaultAttr)
      : baseAttr{baseAttr},
        defaultAttr{defaultAttr} {}
  bool operator==(const KeyTy &key) const {
    return key.first == baseAttr && key.second == defaultAttr;
  }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  static DefaultAttrStorage *construct(AttributeStorageAllocator &alloc,
                                       const KeyTy &key) {
    return new (alloc.allocate<DefaultAttrStorage>())
        DefaultAttrStorage{key.first, key.second};
  }

  Attribute baseAttr;
  Attribute defaultAttr;
};

struct IsaAttrStorage : public AttributeStorage {
  using KeyTy = mlir::SymbolRefAttr;

  explicit IsaAttrStorage(mlir::SymbolRefAttr symRef)
      : symRef{symRef} {}

  bool operator==(const KeyTy &key) const { return key == symRef; }
  static llvm::hash_code hashKey(const KeyTy &key) { return hash_value(key); }

  static IsaAttrStorage *construct(AttributeStorageAllocator &alloc,
                                   const KeyTy &key) {
    return new (alloc.allocate<IsaAttrStorage>()) IsaAttrStorage{key};
  }

  mlir::SymbolRefAttr symRef;

  StringRef getDialectRef() { return symRef.getRootReference(); }
  StringRef getAttrRef() { return symRef.getLeafReference(); }
};

struct AttrComparator {
  bool operator()(Attribute lhs, Attribute rhs) const {
    return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
  }
};

} // end namespace detail

/// ElementsOf implementation.
ElementsOfAttr ElementsOfAttr::get(Type elTy) {
  return Base::get(elTy.getContext(), Kind, elTy);
}

LogicalResult ElementsOfAttr::verify(Attribute attr) {
  if (auto elementsAttr = attr.dyn_cast<mlir::ElementsAttr>()) {
    auto baseTy = getImpl()->type;
    auto elTy = elementsAttr.getType().getElementType();
    return SpecTypes::delegateVerify(baseTy, elTy);
  }
  return failure();
}

/// RankedElementsAttr implementation.
RankedElementsAttr RankedElementsAttr::getChecked(Location loc,
                                                  ArrayRef<int64_t> dims) {
  return Base::getChecked(loc, Kind, dims);
}

LogicalResult RankedElementsAttr::verifyConstructionInvariants(
    Location loc, ArrayRef<int64_t> dims) {
  if (llvm::any_of(dims, [](auto i) { return i <= 0; }))
    return emitError(loc, "dimension list must have positive sizes");
  return success();
}

LogicalResult RankedElementsAttr::verify(Attribute attr) {
  if (auto elementsAttr = attr.dyn_cast<mlir::ElementsAttr>()) {
    auto shape = elementsAttr.getType();
    return success(shape.hasRank() && shape.getShape() == getImpl()->dims);
  }
  return failure();
}

/// ArrayOfAttr implementation.
ArrayOfAttr ArrayOfAttr::getChecked(Location loc, Attribute constraint) {
  return Base::getChecked(loc, Kind, constraint);
}

LogicalResult ArrayOfAttr::verifyConstructionInvariants(Location loc,
                                                        Attribute constraint) {
  if (!SpecAttrs::is(constraint))
    return emitError(loc) << "expected an attribute constraint but got a "
        << "concrete attribute: " << constraint;
  return success();
}

LogicalResult ArrayOfAttr::verify(Attribute attr) {
  if (auto arrAttr = attr.dyn_cast<mlir::ArrayAttr>()) {
    return success(llvm::all_of(arrAttr, [&](auto val) {
      return succeeded(SpecAttrs::delegateVerify(getImpl()->attr, val));
    }));
  }
  return failure();
}

/// ConstantAttr implementation.
ConstantAttr ConstantAttr::get(Attribute attr) {
  return Base::get(attr.getContext(), Kind, attr);
}

LogicalResult ConstantAttr::verify(Attribute attr) {
  return success(attr == getImpl()->attr);
}

/// Helper functions.
namespace impl {
static LogicalResult verifyAttrList(Location loc, ArrayRef<Attribute> attrs) {
  if (attrs.empty())
    return emitError(loc) << "attribute list cannot be empty";
  llvm::SmallPtrSet<Attribute, 4> attrSet{std::begin(attrs),
                                          std::end(attrs)};
  if (std::size(attrSet) != std::size(attrs))
    return emitError(loc) << "duplicate attributes passed";
  return success();
}
} // end namespace impl

static auto getSortedAttrs(ArrayRef<Attribute> attrs) {
  return getSortedListOf<detail::AttrComparator>(attrs);
}

/// AnyOfAttr implementation
AnyOfAttr AnyOfAttr::getChecked(Location loc, ArrayRef<Attribute> attrs) {
  return Base::getChecked(loc, Kind, getSortedAttrs(attrs));
}

LogicalResult AnyOfAttr::verifyConstructionInvariants(
    Location loc, ArrayRef<Attribute> attrs) {
  return impl::verifyAttrList(loc, attrs);
}

LogicalResult AnyOfAttr::verify(Attribute attr) {
  for (auto baseAttr : getImpl()->attrs) {
    if (succeeded(SpecAttrs::delegateVerify(baseAttr, attr)))
      return success();
  }
  return failure();
}

/// AllOfAttr implementation
AllOfAttr AllOfAttr::getChecked(Location loc, ArrayRef<Attribute> attrs) {
  return Base::getChecked(loc, Kind, getSortedAttrs(attrs));
}

LogicalResult AllOfAttr::verifyConstructionInvariants(
    Location loc, ArrayRef<Attribute> attrs) {
  return impl::verifyAttrList(loc, attrs);
}

LogicalResult AllOfAttr::verify(Attribute attr) {
  for (auto baseAttr : getImpl()->attrs) {
    if (failed(SpecAttrs::delegateVerify(baseAttr, attr)))
      return failure();
  }
  return success();
}

/// OfTypeAttr implementation.
OfTypeAttr OfTypeAttr::get(Type ty) {
  return Base::get(ty.getContext(), Kind, ty);
}

LogicalResult OfTypeAttr::verify(Attribute attr) {
  return SpecTypes::delegateVerify(getImpl()->type, attr.getType());
}

/// OptionalAttr implementation.
OptionalAttr OptionalAttr::get(Attribute baseAttr) {
  return Base::get(baseAttr.getContext(), Kind, baseAttr);
}

/// TODO assert that this constraint is top-level
LogicalResult OptionalAttr::verify(Attribute attr) {
  if (!attr) // null attribute is acceptable
    return success();
  return SpecAttrs::delegateVerify(getImpl()->attr, attr);
}

/// DefaultAttr implementation.
DefaultAttr DefaultAttr::get(Attribute baseAttr, Attribute defaultAttr) {
  return Base::get(baseAttr.getContext(), Kind, baseAttr, defaultAttr);
}

/// TODO assert that this constraint is top-level
LogicalResult DefaultAttr::verify(Attribute attr) {
  auto baseAttr = getImpl()->baseAttr;
  if (SpecAttrs::is(baseAttr))
    return SpecAttrs::delegateVerify(baseAttr, attr);
  return success(baseAttr == attr);
}

Attribute DefaultAttr::getDefaultValue() {
  return getImpl()->defaultAttr;
}

/// IsaAttr implementation.
IsaAttr IsaAttr::getChecked(Location loc, mlir::SymbolRefAttr attrRef) {
  return Base::getChecked(loc, Kind, attrRef);
}

LogicalResult IsaAttr::verifyConstructionInvariants(
    Location loc, mlir::SymbolRefAttr attrRef) {
  if (!attrRef)
    return emitError(loc) << "Null attribute reference";
  auto nestedRefs = attrRef.getNestedReferences();
  if (llvm::size(nestedRefs) != 1)
    return emitError(loc) << "Expected attribute reference to have depth 2, "
        << "in the form @dialect::@attrname";
  return success();
}

static DynamicAttributeImpl *lookupAttrReference(
    MLIRContext *ctx, StringRef dialectName, StringRef attrName) {
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
  /// TODO attribute alias?
  /// Resolve the attribute reference.
  auto *attrImpl = dynDialect->lookupAttr(attrName);
  if (!attrImpl) {
    llvm::errs() << "error: dialect '" << dialectName
        << "' does not have attribute '" << attrName << '\'';
    return nullptr;
  }
  return attrImpl;
}

LogicalResult IsaAttr::verify(Attribute attr) {
  auto dynAttr = attr.dyn_cast<DynamicAttribute>();
  if (!dynAttr)
    return failure();
  auto *attrImpl = lookupAttrReference(
    getContext(), getImpl()->getDialectRef(), getImpl()->getAttrRef());
  if (!attrImpl)
    return failure();
  return success(attrImpl == dynAttr.getDynImpl());
}

/// Attribute printing.
namespace {

template <typename AttrList>
void printAttrList(const AttrList &attrs, DialectAsmPrinter &printer) {
  printer << '<';
  llvm::interleaveComma(attrs, printer,
                        [&](auto attr) { printer.printAttribute(attr); });
  printer << '>';
}

} // end anonymous namespace

void ElementsOfAttr::print(DialectAsmPrinter &printer) {
  printer << getAttrName() << '<';
  printer.printType(getImpl()->type);
  printer << '>';
}

void RankedElementsAttr::print(DialectAsmPrinter &printer) {
  printer << getAttrName() << '<';
  impl::printIntegerList(printer, getImpl()->dims);
  printer << '>';
}

void ArrayOfAttr::print(DialectAsmPrinter &printer) {
  printer << getAttrName() << '<';
  printer.printAttribute(getImpl()->attr);
  printer << '>';
}

void ConstantAttr::print(DialectAsmPrinter &printer) {
  printer << getAttrName() << '<';
  printer.printAttribute(getImpl()->attr);
  printer << '>';
}

void AnyOfAttr::print(DialectAsmPrinter &printer) {
  printer << getAttrName();
  printAttrList(getImpl()->attrs, printer);
}

void AllOfAttr::print(DialectAsmPrinter &printer) {
  printer << getAttrName();
  printAttrList(getImpl()->attrs, printer);
}

void OfTypeAttr::print(DialectAsmPrinter &printer) {
  printer << getAttrName() << '<';
  printer.printType(getImpl()->type);
  printer << '>';
}

void OptionalAttr::print(DialectAsmPrinter &printer) {
  printer << getAttrName() << '<';
  printer.printAttribute(getImpl()->attr);
  printer << '>';
}

void DefaultAttr::print(DialectAsmPrinter &printer) {
  printer << getAttrName() << '<';
  printer.printAttribute(getImpl()->baseAttr);
  printer << ',';
  printer.printAttribute(getImpl()->defaultAttr);
  printer << '>';
}

void IsaAttr::print(DialectAsmPrinter &printer) {
  printer << getAttrName() << '<';
  printer.printAttribute(getImpl()->symRef);
  printer << '>';
}

} // end namespace dmc
