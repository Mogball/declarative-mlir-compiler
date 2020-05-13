#include "dmc/Spec/SpecOps.h"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Parser.h>
#include <mlir/IR/PatternMatch.h>

using namespace mlir;

namespace dmc {

namespace {

class TypeReparser {
public:
  TypeReparser(OperationOp opOp) : opOp{opOp} {}

  Optional<mlir::DictionaryAttr> reparseAttrs(mlir::DictionaryAttr opAttrs);

  template <typename TypeRange>
  Optional<std::vector<Type>> reparseTypes(TypeRange types, StringRef name);

  LogicalResult reparseTypes();

private:
  OperationOp opOp;
};

Type reparseType(Type type) {
  /// Unresolved dynamic types can be nested inside other types, so, lacking
  /// a good way to generically traverse nested types, we have to print the
  /// whole type and reparse it.
  std::string buf;
  llvm::raw_string_ostream os{buf};
  type.print(os);
  // TODO improve error reporting
  return mlir::parseType(os.str(), type.getContext());
}

Attribute reparseAttr(Attribute attr) {
  std::string buf;
  llvm::raw_string_ostream os{buf};
  attr.print(os);
  return mlir::parseAttribute(os.str(), attr.getContext());
}

template <typename TypeRange>
Optional<std::vector<Type>> TypeReparser::reparseTypes(
    TypeRange types, StringRef name) {
  std::vector<Type> newTypes;
  newTypes.reserve(llvm::size(types));
  unsigned idx = 0;
  for (auto type : types) {
    if (auto newType = reparseType(type)) {
      newTypes.push_back(newType);
    } else {
      opOp.emitOpError("failed to parse type for ") << name << " #" << idx;
      return llvm::None;
    }
    ++idx;
  }
  return newTypes;
}

Optional<mlir::DictionaryAttr>
TypeReparser::reparseAttrs(mlir::DictionaryAttr opAttrs) {
  NamedAttrList newNamedAttrs;
  for (auto [name, attr] : opAttrs.getValue()) {
    if (auto newAttr = reparseAttr(attr)) {
      newNamedAttrs.push_back({name, newAttr});
    } else {
      opOp.emitOpError("failed to parse attribute '") << name << '\'';
      return llvm::None;
    }
  }
  return mlir::DictionaryAttr::get(newNamedAttrs, opAttrs.getContext());
}

LogicalResult TypeReparser::reparseTypes() {
  /// Reparse any place where a custom type may appear: Op type and attributes.
  auto opTy = opOp.getType();
  auto argTys = reparseTypes(opTy.getInputs(), "operand");
  auto resTys = reparseTypes(opTy.getResults(), "result");
  if (!argTys || !resTys)
    return failure();
  auto newOpTy = mlir::FunctionType::get(*argTys, *resTys, opOp.getContext());
  if (newOpTy != opTy)
    opOp.setOpType(newOpTy);

  auto opAttrs = opOp.getOpAttrs();
  auto newOpAttrs = reparseAttrs(opAttrs);
  if (!newOpAttrs)
    return failure();
  if (newOpAttrs != opAttrs)
    opOp.setOpAttrs(*newOpAttrs);

  return success();
}

} // end anonymous namespace


LogicalResult lowerOpaqueTypes(DialectOp dialectOp) {
  for (auto opOp : dialectOp.getOps<OperationOp>()) {
    TypeReparser lower{opOp};
    if (failed(lower.reparseTypes()))
      return failure();
  }
  return success();
}

} // end namespace dmc
