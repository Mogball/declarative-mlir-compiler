#include "dmc/Spec/SpecOps.h"

#include <mlir/Parser.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;
using namespace mlir::dmc;

namespace dmc {

/// TODO Improve error messages for string buffers.
namespace impl {
template <typename T, typename ReparseFn>
auto reparseSymbol(T arg, ReparseFn &&reparseFn) {
  std::string buf;
  llvm::raw_string_ostream os{buf};
  arg.print(os);
  return reparseFn(os.str());
}

Type reparseType(Type type) {
  return reparseSymbol(type, [&](StringRef data)
                     { return parseType(data, type.getContext()); });
}

Attribute reparseAttr(Attribute attr) {
  return reparseSymbol(attr, [&](StringRef data)
                     { return parseAttribute(data, attr.getType()); });
}
} // end namespace impl

/// OperationOp reparsing.
namespace {
template <typename TypeRange, typename TypeContainerT>
ParseResult reparseTypeRange(OperationOp op, TypeRange types,
                             TypeContainerT &newTypes, StringRef name) {
  newTypes.reserve(llvm::size(types));
  unsigned idx = 0;
  for (auto type : types) {
    if (auto newType = impl::reparseType(type)) {
      newTypes.push_back(newType);
    } else {
      return op.emitOpError("failed to parse type for ") << name
          << " #" << idx;
    }
    ++idx;
  }
  return success();
}

template <typename NamedAttrRange>
ParseResult reparseNamedAttrs(OperationOp op, NamedAttrRange attrs,
                              NamedAttrList &newAttrs) {
  for (auto [name, attr] : attrs) {
    if (auto newAttr = impl::reparseAttr(attr)) {
      newAttrs.push_back({name, newAttr});
    } else {
      return op.emitOpError("failed to parse attribute '") << name << '\'';
    }
  }
  return success();
}
} // end anonymous namespace

ParseResult OperationOp::reparse() {
  /// Reparse operation type.
  auto opTy = getType();
  std::vector<Type> argTys, resTys;
  if (failed(reparseTypeRange(*this, opTy.getInputs(), argTys, "operand")))
    return failure();
  if (failed(reparseTypeRange(*this, opTy.getResults(), resTys, "result")))
    return failure();
  auto newOpTy = mlir::FunctionType::get(argTys, resTys, getContext());
  if (newOpTy != opTy)
    setOpType(newOpTy);

  /// Reparse operation attributes.
  auto opAttrs = getOpAttrs();
  NamedAttrList attrs;
  if (failed(reparseNamedAttrs(*this, opAttrs.getValue(), attrs)))
    return failure();
  auto newOpAttrs = mlir::DictionaryAttr::get(attrs, getContext());
  if (newOpAttrs != opAttrs)
    setOpAttrs(newOpAttrs);
  return success();
}

/// TypeOp reparsing.
namespace {
template <typename OpT, typename ParamRange, typename ParamListT>
ParseResult reparseParameterList(OpT op, ParamRange params,
                                 ParamListT &newParams) {
  newParams.reserve(llvm::size(params));
  unsigned idx = 0;
  for (auto param : params) {
    if (auto newParam = impl::reparseAttr(param)) {
      newParams.push_back(newParam);
    } else {
      return op.emitOpError("failed to parse parameter #") << idx;
    }
    ++idx;
  }
  return success();
}

ParseResult reparseParameterList(ParameterList op) {
  std::vector<Attribute> newParams;
  if (failed(reparseParameterList(op, op.getParameters(), newParams)))
    return failure();
  op.setParameters(newParams);
  return success();
}
} // end anonymous namespace

ParseResult TypeOp::reparse() {
  /// Reparse parameter list.
  return reparseParameterList(cast<ParameterList>(getOperation()));
}

/// AttributeOp reparsing.
ParseResult AttributeOp::reparse() {
  /// Reparse parameter list.
  return reparseParameterList(cast<ParameterList>(getOperation()));
}

} // end namespace dmc
