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

/// TypedAttrStorage implementation.
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

/// ConstantAttrStorage implementation.
struct ConstantAttrStorage : public AttributeStorage {
  using KeyTy = Attribute;

  explicit ConstantAttrStorage(KeyTy key) : attr{key} {}
  bool operator==(const KeyTy &key) const { return key == attr; }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return hash_value(key);
  }

  static ConstantAttrStorage *construct(AttributeStorageAllocator &alloc,
                                        const KeyTy &key) {
    return new (alloc.allocate<ConstantAttrStorage>())
        ConstantAttrStorage{key};
  }

  KeyTy attr;
};

/// AttrListStorage implementation.
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

struct DefaultAttrStorage : public AttributeStorage {
  /// Store the base Attribute constraint and the default value.
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

/// ConstantAttr implementation.
ConstantAttr ConstantAttr::get(Attribute attr) {
  return Base::get(attr.getContext(), SpecAttrs::Constant, attr);
}

ConstantAttr ConstantAttr::getChecked(Location loc, Attribute attr) {
  return Base::getChecked(loc, SpecAttrs::Constant, attr);
}

LogicalResult ConstantAttr::verifyConstructionInvariants(
    Location loc, Attribute attr) {
  if (!attr)
    return emitError(loc) << "Attribute cannot be null";
  return success();
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
AnyOfAttr AnyOfAttr::get(ArrayRef<Attribute> attrs) {
  auto *ctx = attrs.front().getContext();
  return Base::get(ctx, SpecAttrs::AnyOf, getSortedAttrs(attrs));
}

AnyOfAttr AnyOfAttr::getChecked(Location loc, ArrayRef<Attribute> attrs) {
  return Base::getChecked(loc, SpecAttrs::AnyOf, getSortedAttrs(attrs));
}

LogicalResult AnyOfAttr::verifyConstructionInvariants(
    Location loc, ArrayRef<Attribute> attrs) {
  return impl::verifyAttrList(loc, attrs);
}

LogicalResult AnyOfAttr::verify(Attribute attr) {
  for (auto baseAttr : getImpl()->attrs) {
    if (SpecAttrs::is(baseAttr) &&
        succeeded(SpecAttrs::delegateVerify(baseAttr, attr)))
      return success();
    else if (baseAttr == attr)
      return success();
  }
  return failure();
}

/// AllOfAttr implementation
AllOfAttr AllOfAttr::get(ArrayRef<Attribute> attrs) {
  auto *ctx = attrs.front().getContext();
  return Base::get(ctx, SpecAttrs::AllOf, getSortedAttrs(attrs));
}

AllOfAttr AllOfAttr::getChecked(Location loc, ArrayRef<Attribute> attrs) {
  return Base::getChecked(loc, SpecAttrs::AllOf, getSortedAttrs(attrs));
}

LogicalResult AllOfAttr::verifyConstructionInvariants(
    Location loc, ArrayRef<Attribute> attrs) {
  return impl::verifyAttrList(loc, attrs);
}

LogicalResult AllOfAttr::verify(Attribute attr) {
  for (auto baseAttr : getImpl()->attrs) {
    if (SpecAttrs::is(baseAttr) &&
        failed(SpecAttrs::delegateVerify(baseAttr, attr)))
      return failure();
    else if (baseAttr != attr)
      return failure();
  }
  return success();
}

/// OfTypeAttr implementation.
OfTypeAttr OfTypeAttr::get(Type ty) {
  return Base::get(ty.getContext(), SpecAttrs::OfType, ty);
}

OfTypeAttr OfTypeAttr::getChecked(Location loc, Type ty) {
  return Base::getChecked(loc, SpecAttrs::OfType, ty);
}

LogicalResult OfTypeAttr::verifyConstructionInvariants(Location loc, Type ty) {
  if (!ty)
    return emitError(loc) << "type cannot be null";
  return success();
}

LogicalResult OfTypeAttr::verify(Attribute attr) {
  if (!attr)
    return failure();
  auto baseTy = getImpl()->type;
  if (SpecTypes::is(baseTy))
    return SpecTypes::delegateVerify(baseTy, attr.getType());
  return success(baseTy == attr.getType());
}

/// OptionalAttr implementation.
OptionalAttr OptionalAttr::get(Attribute baseAttr) {
  return Base::get(baseAttr.getContext(), SpecAttrs::Optional, baseAttr);
}

OptionalAttr OptionalAttr::getChecked(Location loc, Attribute baseAttr) {
  return Base::getChecked(loc, SpecAttrs::Optional, baseAttr);
}

LogicalResult OptionalAttr::verifyConstructionInvariants(
    Location loc, Attribute baseAttr) {
  /// TODO assert that this constraint is top-level
  if (!baseAttr)
    return emitError(loc) << "base attribute cannot be null";
  return success();
}

LogicalResult OptionalAttr::verify(Attribute attr) {
  if (!attr) // null attribute is acceptable
    return success();
  auto baseAttr = getImpl()->attr;
  if (SpecAttrs::is(baseAttr))
    return SpecAttrs::delegateVerify(baseAttr, attr);
  return success(baseAttr == attr);
}

/// DefaultAttr implementation.
DefaultAttr DefaultAttr::get(Attribute baseAttr, Attribute defaultAttr) {
  return Base::get(baseAttr.getContext(), SpecAttrs::Default,
                   baseAttr, defaultAttr);
}

DefaultAttr DefaultAttr::getChecked(Location loc, Attribute baseAttr,
                                    Attribute defaultAttr) {
  return Base::getChecked(loc, SpecAttrs::Default, baseAttr, defaultAttr);
}

LogicalResult DefaultAttr::verifyConstructionInvariants(
    Location loc, Attribute baseAttr, Attribute defaultAttr) {
  /// TODO assert that this constraint is top-level
  if (!baseAttr)
    return emitError(loc) << "attribute constraint cannot be null";
  if (!defaultAttr)
    return emitError(loc) << "attribute default value cannot be null";
  return success();
}

LogicalResult DefaultAttr::verify(Attribute attr) {
  auto baseAttr = getImpl()->baseAttr;
  if (SpecAttrs::is(baseAttr))
    return SpecAttrs::delegateVerify(baseAttr, attr);
  return success(baseAttr == attr);
}

/// IsaAttr implementation.
IsaAttr IsaAttr::get(mlir::SymbolRefAttr attrRef) {
  return Base::get(attrRef.getContext(), SpecAttrs::Isa, attrRef);
}

IsaAttr IsaAttr::getChecked(Location loc, mlir::SymbolRefAttr attrRef) {
  return Base::getChecked(loc, SpecAttrs::Isa, attrRef);
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
  return success(attrImpl == dynAttr.getAttrImpl());
}

/// Attribute printing.
namespace {

template <typename AttrList>
void printAttrList(const AttrList &attrs, DialectAsmPrinter &printer) {
  printer << '<';
  auto it = std::begin(attrs);
  printer.printAttribute(*it++);
  for (auto e = std::end(attrs); it != e; ++it) {
    printer << ',';
    printer.printAttribute(*it);
  }
  printer << '>';
}

} // end anonymous namespace

void SpecDialect::printAttribute(
    Attribute attr, DialectAsmPrinter &printer) const {
  PrintAction<DialectAsmPrinter> action{printer};
  SpecAttrs::kindSwitch(action, attr);
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
