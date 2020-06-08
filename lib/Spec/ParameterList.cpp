#include "dmc/Spec/ParameterList.h"
#include "dmc/Spec/SpecAttrs.h"
#include "dmc/Spec/Parsing.h"

using namespace dmc;

namespace mlir {
namespace dmc {

#include "dmc/Spec/ParameterList.cpp.inc"

namespace detail {
struct NamedParameterStorage : public AttributeStorage {
  using KeyTy = std::pair<StringRef, Attribute>;

  explicit NamedParameterStorage(StringRef name, Attribute constraint)
      : name{name}, constraint{constraint} {}

  bool operator==(const KeyTy &key) const {
    return key.first == name && key.second == constraint;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  static NamedParameterStorage *construct(AttributeStorageAllocator &alloc,
                                          const KeyTy &key) {
    return new (alloc.allocate<NamedParameterStorage>())
        NamedParameterStorage{alloc.copyInto(key.first), key.second};
  }

  StringRef name;
  Attribute constraint;
};
} // end namespace detail

NamedParameter NamedParameter::get(StringRef name, Attribute constraint) {
  return Base::get(constraint.getContext(), ::dmc::AttrKinds::NamedParameter,
                   name, constraint);
}

NamedParameter NamedParameter::getChecked(Location loc, StringRef name,
                                          Attribute constraint) {
  return Base::getChecked(loc, ::dmc::AttrKinds::NamedParameter, name,
                          constraint);
}

LogicalResult NamedParameter::verifyConstructionInvariants(
    Location loc, StringRef name, Attribute constraint) {
  if (!::dmc::SpecAttrs::is(constraint) && !constraint.isa<OpaqueAttr>())
    return emitError(loc) << "expected a valid attribute constraint";
  return success();
}

StringRef NamedParameter::getName() {
  return getImpl()->name;
}

Attribute NamedParameter::getConstraint() {
  return getImpl()->constraint;
}

ParseResult ParameterList::parse(OpAsmParser &parser,
                                 NamedAttrList &attrList) {
  SmallVector<Attribute, 2> params;
  if (succeeded(parser.parseLess())) {
    StringRef name;
    Attribute constraint;
    do {
      auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
      if (parser.parseKeyword(&name) || parser.parseColon() ||
          ::dmc::impl::parseSingleAttribute(parser, constraint))
        return failure();
      params.push_back(NamedParameter::getChecked(loc, name, constraint));
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseGreater())
      return failure();
  }
  attrList.append(Trait<int>::getParametersAttrName(),
                  parser.getBuilder().getArrayAttr(params));
  return success();
}

} // end namespace dmc
} // end namespace mlir
