#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Dynamic/DynamicType.h"
#include "dmc/Dynamic/DynamicAttribute.h"
#include "dmc/Embed/Init.h"

#include <llvm/ADT/StringMap.h>
#include <mlir/IR/Operation.h>

using namespace mlir;

namespace dmc {

class DynamicContext::Impl {
  friend class DynamicContext;

  /// A registry of symbols and their associated dynamic dialect.
  DenseMap<const void *, DynamicDialect *> dialectSymbols;

  template <typename SymbolT> DynamicDialect *lookupDialectFor(SymbolT sym) {
    auto it = dialectSymbols.find(sym.getAsOpaquePointer());
    return it == std::end(dialectSymbols) ? nullptr : it->second;
  }

  template <typename SymbolT>
  LogicalResult registerDialectSymbol(DynamicDialect *dialect, SymbolT sym) {
    auto [it, inserted] = dialectSymbols.try_emplace(sym.getAsOpaquePointer(),
                                                     dialect);
    return success(inserted);
  }
};

DynamicContext::~DynamicContext() = default;

DynamicContext::DynamicContext(MLIRContext *ctx)
    : Dialect{getDialectNamespace(), ctx, TypeID::get<DynamicContext>()},
      typeIdAlloc{getFixedTypeIDAllocator()},
      impl{std::make_unique<Impl>()} {
  // Automatically initialize the interpreter
  py::init(ctx);
}

DynamicDialect *DynamicContext::createDynamicDialect(StringRef name) {
  auto *dialect = new DynamicDialect{name, this};
  auto typeId = dynamic_cast<DynamicObject *>(dialect)->getTypeID();
  auto ctor = [dialect, typeId]() {
    std::unique_ptr<DynamicDialect> ptr{dialect};
    ptr->dialectID = typeId;
    return ptr;
  };
  getContext()->getOrCreateDialect(name, typeId, ctor);
  return dialect;
}

DynamicDialect *DynamicContext::lookupDialectFor(Type type) {
  if (auto dynTy = type.dyn_cast<DynamicType>())
    return dynTy.getDynImpl()->getDialect();
  return impl->lookupDialectFor(type);
}

DynamicDialect *DynamicContext::lookupDialectFor(Attribute attr) {
  if (auto dynAttr = attr.dyn_cast<DynamicAttribute>())
    return dynAttr.getDynImpl()->getDialect();
  return impl->lookupDialectFor(attr);
}

DynamicDialect *DynamicContext::lookupDialectFor(OperationName opName) {
  return impl->lookupDialectFor(opName);
}

LogicalResult DynamicContext::registerDialectSymbol(DynamicDialect *dialect,
                                                    Type type) {
  return impl->registerDialectSymbol(dialect, type);
}

LogicalResult DynamicContext::registerDialectSymbol(DynamicDialect *dialect,
                                                    Attribute attr) {
  return impl->registerDialectSymbol(dialect, attr);
}

LogicalResult DynamicContext::registerDialectSymbol(DynamicDialect *dialect,
                                                    OperationName opName) {
  return impl->registerDialectSymbol(dialect, opName);
}

} // end namespace dmc
