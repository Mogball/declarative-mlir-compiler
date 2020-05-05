#include "dmc/Spec/SpecAttrs.h"
#include "dmc/Spec/SpecDialect.h"
#include "dmc/Spec/Support.h"
#include "dmc/Spec/SpecTypeDetail.h"
#include "dmc/Spec/SpecTypeImplementation.h"

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

TypedAttrStorage *TypedAttrStorage::construct(
    AttributeStorageAllocator &alloc, const KeyTy &key) {
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

  static ConstantAttrStorage *construct(
      AttributeStorageAllocator &alloc, const KeyTy &key) {
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

  static AttrListStorage *construct(
      AttributeStorageAllocator &alloc, KeyTy key) {
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

  static OneTypeAttrStorage *construct(
      AttributeStorageAllocator &alloc, KeyTy key) {
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

  static DefaultAttrStorage *construct(
      AttributeStorageAllocator &alloc, const KeyTy &key) {
    return new (alloc.allocate<DefaultAttrStorage>())
        DefaultAttrStorage{key.first, key.second};
  }

  Attribute baseAttr;
  Attribute defaultAttr;
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
OfTypeAttr OfTypeAttr::get(Type ty, MLIRContext *ctx) {
  return Base::get(ctx, SpecAttrs::OfType, ty);
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

/// Attribute printing.
namespace {

template <typename AttrT>
auto *getTypeImpl(Attribute attr) {
  return attr.template cast<AttrT>().getImpl()
      ->type.template cast<typename AttrT::Underlying>().getImpl();
}

void printAttrList(detail::AttrListStorage *impl,
                   DialectAsmPrinter &printer) {
  printer << '<';
  auto &attrs = impl->attrs;
  auto it = std::begin(attrs);
  printer.printAttribute(*it++);
  for (auto e = std::end(attrs); it != e; ++it) {
    printer << ',';
    printer.printAttribute(*it);
  }
  printer << '>';
}

} // end anonymous namespace

void SpecDialect::printAttribute(Attribute attr,
    DialectAsmPrinter &printer) const {
  using namespace SpecAttrs;
  assert(is(attr) && "Not a SpecAttr");
  switch (attr.getKind()) {
  case Any:
    printer << "Any";
    break;
  case Bool:
    printer << "Bool";
    break;
  case Index:
    printer << "Index";
    break;
  case APInt:
    printer << "APInt";
    break;
  case AnyI:
    printer << "AnyI";
    printSingleWidth(getTypeImpl<AnyIAttr>(attr), printer);
    break;
  case I:
    printer << "I";
    printSingleWidth(getTypeImpl<IAttr>(attr), printer);
    break;
  case SI:
    printer << "SI";
    printSingleWidth(getTypeImpl<SIAttr>(attr), printer);
    break;
  case UI:
    printer << "UI";
    printSingleWidth(getTypeImpl<UIAttr>(attr), printer);
    break;
  case F:
    printer << "F";
    printSingleWidth(getTypeImpl<FAttr>(attr), printer);
    break;
  case String:
    printer << "String";
    break;
  case Type:
    printer << "Type";
    break;
  case Unit:
    printer << "Unit";
    break;
  case Dictionary:
    printer << "Dictionary";
    break;
  case Elements:
    printer << "Elements";
    break;
  case Array:
    printer << "Array";
    break;
  case SymbolRef:
    printer << "SymbolRef";
    break;
  case FlatSymbolRef:
    printer << "FlatSymbolRef";
    break;
  case Constant:
    attr.cast<ConstantAttr>().print(printer);
    break;
  case AnyOf:
    printer << "AnyOf";
    printAttrList(attr.cast<AnyOfAttr>().getImpl(), printer);
    break;
  case AllOf:
    printer << "AllOf";
    printAttrList(attr.cast<AllOfAttr>().getImpl(), printer);
    break;
  case OfType:
    attr.cast<OfTypeAttr>().print(printer);
    break;
  case Optional:
    attr.cast<OptionalAttr>().print(printer);
    break;
  case Default:
    attr.cast<DefaultAttr>().print(printer);
    break;
  default:
    llvm_unreachable("Unknown SpecAttr");
    break;
  }
}

void ConstantAttr::print(DialectAsmPrinter &printer) {
  printer << "Constant<";
  printer.printAttribute(getImpl()->attr);
  printer << '>';
}

void OfTypeAttr::print(DialectAsmPrinter &printer) {
  printer << "OfType<";
  printer.printType(getImpl()->type);
  printer << '>';
}

void OptionalAttr::print(DialectAsmPrinter &printer) {
  printer << "Optional<";
  printer.printAttribute(getImpl()->attr);
  printer << '>';
}

void DefaultAttr::print(DialectAsmPrinter &printer) {
  printer << "Default<";
  printer.printAttribute(getImpl()->baseAttr);
  printer << ',';
  printer.printAttribute(getImpl()->defaultAttr);
  printer << '>';
}

} // end namespace dmc
