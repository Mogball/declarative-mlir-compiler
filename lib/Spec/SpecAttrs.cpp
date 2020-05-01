#include "dmc/Spec/SpecAttrs.h"
#include "dmc/Spec/SpecDialect.h"
#include "dmc/Spec/Support.h"
#include "dmc/Spec/SpecTypeDetail.h"

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
namespace {

namespace impl {

LogicalResult verifyAttrList(Location loc, ArrayRef<Attribute> attrs) {
  if (attrs.empty())
    return emitError(loc) << "attribute list cannot be empty";
  llvm::SmallPtrSet<Attribute, 4> attrSet{std::begin(attrs), 
                                          std::end(attrs)};
  if (std::size(attrSet) != std::size(attrs))
    return emitError(loc) << "duplicate attributes passed";
  return success();
}
  
} // end namespace impl

auto getSortedAttrs(ArrayRef<Attribute> attrs) {
  return getSortedListOf<detail::AttrComparator>(attrs);
}

} // end anonymous namespace

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

} // end namespace dmc
