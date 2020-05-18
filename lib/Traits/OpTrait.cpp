#include "dmc/Traits/OpTrait.h"
#include "dmc/Traits/Registry.h"
#include "dmc/Spec/Parsing.h"

#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;

namespace dmc {

namespace detail {
struct OpTraitStorage : public AttributeStorage {
  using KeyTy = std::pair<mlir::FlatSymbolRefAttr, mlir::ArrayAttr>;

  explicit OpTraitStorage(const KeyTy &key)
      : name{key.first},
        params{key.second} {}

  bool operator==(const KeyTy &key) const {
    return key.first == name && key.second == params;
  }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  static OpTraitStorage *construct(AttributeStorageAllocator &alloc,
                                   const KeyTy &key) {
    return new (alloc.allocate<OpTraitStorage>()) OpTraitStorage{key};
  }

  mlir::FlatSymbolRefAttr name;
  mlir::ArrayAttr params;
};

struct OpTraitsStorage : public AttributeStorage {
  using KeyTy = mlir::ArrayAttr;

  explicit OpTraitsStorage(KeyTy key) : traits{key} {}
  bool operator==(const KeyTy &key) const { return key == traits; }
  static llvm::hash_code hashKey(const KeyTy &key) { return hash_value(key); }

  static OpTraitsStorage *construct(AttributeStorageAllocator &alloc,
                                    KeyTy key) {
    return new (alloc.allocate<OpTraitsStorage>()) OpTraitsStorage{key};
  }

  KeyTy traits;
};
} // end namespace detail

OpTraitAttr OpTraitAttr::get(mlir::FlatSymbolRefAttr nameAttr,
                             mlir::ArrayAttr paramAttr) {
  return Base::get(nameAttr.getContext(), TraitAttr::OpTrait, nameAttr,
                   paramAttr);
}

OpTraitAttr OpTraitAttr::getChecked(
    mlir::Location loc, mlir::FlatSymbolRefAttr nameAttr,
    mlir::ArrayAttr paramAttr) {
  return Base::getChecked(loc, TraitAttr::OpTrait, nameAttr, paramAttr);
}

LogicalResult OpTraitAttr::verifyConstructionInvariants(
    mlir::Location loc, mlir::FlatSymbolRefAttr nameAttr,
    mlir::ArrayAttr paramAttr) {
  if (!nameAttr)
    return emitError(loc) << "op trait name cannot be null";
  if (!paramAttr)
    return emitError(loc) << "op trait parameter list cannot be null";
  /// Check that the op trait exists.
  auto *registry = loc.getContext()->getRegisteredDialect<TraitRegistry>();
  assert(registry && "TraitRegistry dialect was not registered");
  if (!registry->lookupTrait(nameAttr.getValue()))
    return emitError(loc) << "op trait " << nameAttr << " not found";
  return success();
}

StringRef OpTraitAttr::getName() {
  return getImpl()->name.getValue();
}

ArrayRef<Attribute> OpTraitAttr::getParameters() {
  return getImpl()->params.getValue();
}

OpTraitsAttr OpTraitsAttr::get(mlir::ArrayAttr traits){
  return Base::get(traits.getContext(), TraitAttr::OpTraits, traits);
}

OpTraitsAttr OpTraitsAttr::getChecked(Location loc, mlir::ArrayAttr traits) {
  return Base::getChecked(loc, TraitAttr::OpTraits, traits);
}

LogicalResult OpTraitsAttr::verifyConstructionInvariants(
    Location loc, mlir::ArrayAttr traits) {
  if (!traits)
    return emitError(loc) << "op traits list cannot be null";
  for (auto trait : traits) {
    if (!trait.isa<OpTraitAttr>())
      return emitError(loc) << "op traits list must only contain op traits";
  }
  return success();
}

ArrayRef<Attribute> OpTraitsAttr::getUnderlyingValue() {
  return getImpl()->traits.getValue();
}

Attribute TraitRegistry::parseAttribute(DialectAsmParser &parser,
                                        Type type) const {
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (type) {
    emitError(loc, "unexpected attribute type");
    return {};
  }

  StringRef attrName;
  if (parser.parseKeyword(&attrName))
    return {};
  auto kind = llvm::StringSwitch<TraitAttr::Kinds>(attrName)
      .Case("OpTrait", TraitAttr::OpTrait)
      .Case("OpTraits", TraitAttr::OpTraits)
      .Default(TraitAttr::NUM_ATTRS);

  switch (kind) {
  case TraitAttr::OpTrait: {
    /// Parse OpTrait attribute.
    mlir::FlatSymbolRefAttr nameAttr;
    mlir::ArrayAttr paramAttr;
    if (parser.parseAttribute(nameAttr) ||
        impl::parseOptionalParameterList(parser, paramAttr))
      return {};
    return OpTraitAttr::getChecked(loc, nameAttr, paramAttr);
  }
  case TraitAttr::OpTraits: {
    /// Parse OpTraits attribute.
    mlir::ArrayAttr traitArr;
    if (parser.parseAttribute(traitArr))
      return {};
    return OpTraitsAttr::getChecked(loc, traitArr);
  }
  default:
    emitError(loc, "unknown attribute name");
    return {};
  }
}

void TraitRegistry::printAttribute(Attribute attr,
                                   DialectAsmPrinter &printer) const {
  switch (attr.getKind()) {
  case TraitAttr::OpTrait:
    break;
  case TraitAttr::OpTraits:
    break;
  default:
    llvm_unreachable("Unknown TraitRegistry attribute kind");
    break;
  }
}

} // end namespace dmc
