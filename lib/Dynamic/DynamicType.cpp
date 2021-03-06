#include "dmc/Dynamic/Alias.h"
#include "dmc/Dynamic/DynamicType.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Dynamic/DynamicAttribute.h"
#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Spec/SpecAttrImplementation.h"
#include "dmc/Embed/ParserPrinter.h"

#include <mlir/IR/Location.h>
#include <mlir/IR/Diagnostics.h>

using namespace mlir;

namespace dmc {

/// DynamicTypeStorage stores a reference to the backing DynamicTypeImpl,
/// which conveniently acts as a discriminator between different DynamicType
/// "classes", and the parameter values.
namespace detail {
struct DynamicTypeStorage : public TypeStorage {
  /// Compound key with the Impl instance and the parameter values.
  using KeyTy = std::pair<DynamicTypeImpl *, ArrayRef<Attribute>>;

  explicit DynamicTypeStorage(DynamicTypeImpl *impl, ArrayRef<Attribute> params)
      : impl{impl}, params{params} {}

  /// Compare implmentation pointer and parameter values.
  bool operator==(const KeyTy &key) const {
    return impl == key.first && params == key.second;
  }

  /// Hash combine the implementation pointer and the parameter values.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  /// Create the DynamicTypeStorage.
  static DynamicTypeStorage *construct(TypeStorageAllocator &alloc,
                                       const KeyTy &key) {
    return new (alloc.allocate<DynamicTypeStorage>())
        DynamicTypeStorage{key.first, alloc.copyInto(key.second)};
  }

  /// Pointer to implmentation.
  DynamicTypeImpl *impl;
  /// Store the parameters.
  ArrayRef<Attribute> params;
};
} // end namespace detail

DynamicTypeImpl::DynamicTypeImpl(DynamicDialect *dialect, StringRef name,
                                 NamedParameterRange paramSpec)
    : DynamicObject{dialect->getDynContext()},
      TypeMetadata{name, llvm::None},
      dialect{dialect},
      paramSpec{paramSpec} {
  dialect->addType(getTypeID(), AbstractType::get<DynamicType>(*dialect));
}

Type DynamicTypeImpl::parseType(Location loc, DialectAsmParser &parser) {
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
  return DynamicType::getChecked(loc, this, params);
}

void DynamicTypeImpl::printType(Type type, DialectAsmPrinter &printer) {
  auto dynTy = type.cast<DynamicType>();

  /// Try a formated printer.
  if (printerFcn) {
    py::execPrinter(*printerFcn, printer, dynTy);
    return;
  }

  /// Generic dynamic type printer.
  printer << getName();
  auto params = dynTy.getParams();
  if (!params.empty()) {
    auto it = std::begin(params);
    printer << '<' << (*it++);
    for (auto e = std::end(params); it != e; ++it)
      printer << ',' << (*it);
    printer << '>';
  }
}

void DynamicTypeImpl::setFormat(std::string parserName,
                                std::string printerName) {
  parserFcn = std::move(parserName);
  printerFcn = std::move(printerName);
}

/// One instance of DynamicType needs to be registered for each DynamicDialect,
/// but that isn't possible, so we have to avoid any calls that use the TypeID
/// of DynamicType.
///
/// If DynamicType is registered to each DynamicDialect, MLIR will complain
/// about duplicate symbol registration if more than one DynamicDialect is
/// instantiated. If it is not registered, then Base::get will fail to lookup
/// the Dialect, so we must directly provide the dialect.
DynamicType DynamicType::get(DynamicTypeImpl *impl,
                             ArrayRef<Attribute> params) {
  auto *ctx = impl->getDynContext()->getContext();
  return ctx->getTypeUniquer().get<Base::ImplType>(
      impl->getTypeID(),
      [impl, ctx](TypeStorage *storage) {
        storage->initialize(AbstractType::lookup(impl->getTypeID(), ctx));
      },
      DynamicTypeKind, impl, params);
}

DynamicType DynamicType::getChecked(Location loc, DynamicTypeImpl *impl,
                                    ArrayRef<Attribute> params) {
  if (failed(verifyConstructionInvariants(loc, impl, params)))
    return {};
  return get(impl, params);
}

LogicalResult DynamicType::verifyConstructionInvariants(
    Location loc, DynamicTypeImpl *impl, ArrayRef<Attribute> params) {
  if (impl == nullptr)
    return emitError(loc) << "Null DynamicTypeImpl";
  if (llvm::size(impl->paramSpec) != llvm::size(params))
    return emitError(loc) << "type construction failed: expected "
         << llvm::size(impl->paramSpec) << " parameters but got "
         << llvm::size(params);
  /// Verify that the provided parameters satisfy the dynamic type spec.
  unsigned idx = 0;
  for (auto [spec, param] : llvm::zip(impl->paramSpec, params)) {
    if (failed(SpecAttrs::delegateVerify(spec.getConstraint(), param)))
      return emitError(loc) << "type construction failed: parameter #"
          << idx << " expected " << spec.getConstraint() << " but got "
          << param;
    ++idx;
  }
  return success();
}

DynamicTypeImpl *DynamicType::getDynImpl() {
  return getImpl()->impl;
}

ArrayRef<Attribute> DynamicType::getParams() {
  return getImpl()->params;
}

Attribute DynamicType::getParam(StringRef name) {
  for (auto [spec, param] : llvm::zip(getDynImpl()->getParamSpec(),
                                      getParams())) {
    if (spec.getName() == name)
      return param;
  }
  return {};
}

DynamicType buildDynamicType(StringRef dialectName, StringRef typeName,
                             ArrayRef<Attribute> params, Location loc) {
  auto *dialect = loc.getContext()->getRegisteredDialect(dialectName);
  auto *dynDialect = dynamic_cast<DynamicDialect *>(dialect);
  auto *impl = dynDialect->lookupType(typeName);
  return DynamicType::getChecked(loc, impl, params);
}

Type getAliasedType(StringRef dialectName, StringRef typeName,
                    MLIRContext *ctx) {
  auto *dialect = ctx->getRegisteredDialect(dialectName);
  auto *dynDialect = dynamic_cast<DynamicDialect *>(dialect);
  auto *alias = dynDialect->lookupTypeAlias(typeName);
  return alias->getAliasedType();
}

} // end namespace dmc
