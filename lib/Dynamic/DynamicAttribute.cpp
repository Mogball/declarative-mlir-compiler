#include "dmc/Dynamic/DynamicAttribute.h"
#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Spec/SpecAttrImplementation.h"
#include "dmc/Embed/ParserPrinter.h"

#include <mlir/IR/Location.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/StandardTypes.h>

using namespace mlir;

namespace dmc {

/// DynamicAttributeStorage stores a reference to the DynamicAttribute
/// class descriptor.
namespace detail {
struct DynamicAttributeStorage : public AttributeStorage {
  using KeyTy = std::pair<DynamicAttributeImpl *, ArrayRef<Attribute>>;

  explicit DynamicAttributeStorage(DynamicAttributeImpl *impl,
                                   ArrayRef<Attribute> params)
      : impl{impl}, params{params} {}

  bool operator==(const KeyTy &key) const {
    return impl == key.first && params == key.second;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  static DynamicAttributeStorage *construct(AttributeStorageAllocator &alloc,
                                            const KeyTy &key) {
    return new (alloc.allocate<DynamicAttributeStorage>())
        DynamicAttributeStorage{key.first, alloc.copyInto(key.second)};
  }

  DynamicAttributeImpl *impl;
  ArrayRef<Attribute> params;
};
} // end namespace detail

DynamicAttributeImpl::DynamicAttributeImpl(
    DynamicDialect *dialect, StringRef name, NamedParameterRange paramSpec)
    : DynamicObject{dialect->getDynContext()},
      AttributeMetadata{name, llvm::None, Type{}},
      dialect{dialect},
      paramSpec{paramSpec} {
  dialect->addAttribute(getTypeID(),
                        AbstractAttribute::get<DynamicAttribute>(*dialect));
}

Attribute DynamicAttributeImpl::parseAttribute(Location loc,
                                               DialectAsmParser &parser) {
  std::vector<Attribute> params;
  if (parserFcn) {
    if (!py::execParser(*parserFcn, parser, params))
      return {};
  } else if (!parser.parseOptionalLess()) {
    do {
      Attribute attr;
      if (parser.parseAttribute(attr))
        return {};
      params.push_back(attr);
    } while (!parser.parseOptionalComma());
    if (parser.parseGreater())
      return {};
  }
  return DynamicAttribute::getChecked(loc, this, params);
}

void DynamicAttributeImpl::printAttribute(Attribute attr,
                                          DialectAsmPrinter &printer) {
  auto dynAttr = attr.cast<DynamicAttribute>();

  /// Try a formated printer.
  if (printerFcn) {
    py::execPrinter(*printerFcn, printer, dynAttr);
    return;
  }

  /// Generic dynamic attribute printer.
  printer << getName();
  auto params = dynAttr.getParams();
  if (!params.empty()) {
    auto it = std::begin(params);
    printer << '<' << (*it++);
    for (auto e = std::end(params); it != e; ++it)
      printer << ',' << (*it);
    printer << '>';
  }
}

void DynamicAttributeImpl::setFormat(std::string parserName,
                                     std::string printerName) {
  parserFcn = std::move(parserName);
  printerFcn = std::move(printerName);
}

/// Since dynamic attributes are not registered with a Dialect or the MLIR
/// context, we need to directly call the Attribute uniquer.
DynamicAttribute DynamicAttribute::get(DynamicAttributeImpl *impl,
                                       ArrayRef<Attribute> params) {
  return Base::get(impl->getDynContext()->getContext(), DynamicAttributeKind,
                   impl, params);
}

DynamicAttribute DynamicAttribute::getChecked(
    Location loc, DynamicAttributeImpl *impl, ArrayRef<Attribute> params) {
  return Base::getChecked(loc, DynamicAttributeKind, impl, params);
}

LogicalResult DynamicAttribute::verifyConstructionInvariants(
  Location loc, DynamicAttributeImpl *impl, ArrayRef<Attribute> params) {
  if (impl == nullptr)
    return emitError(loc) << "Null DynamicAttributeImpl";
  if (llvm::size(impl->paramSpec) != llvm::size(params))
    return emitError(loc) << "attribute construction failed: expected "
        << llvm::size(impl->paramSpec) << " parameters but got "
        << llvm::size(params);
  unsigned idx = 0;
  for (auto [spec, param] : llvm::zip(impl->paramSpec, params)) {
    if (failed(SpecAttrs::delegateVerify(spec.getConstraint(), param)))
      return emitError(loc) << "attribute construction failed: parameter #"
           << idx << " expected " << spec.getConstraint() << " but got "
           << param;
    ++idx;
  }
  return success();
}

DynamicAttributeImpl *DynamicAttribute::getDynImpl() {
  return getImpl()->impl;
}

ArrayRef<Attribute> DynamicAttribute::getParams() {
  return getImpl()->params;
}

} // end namespac dmc
