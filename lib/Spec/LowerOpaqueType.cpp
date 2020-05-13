#include "dmc/Spec/SpecOps.h"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Parser.h>
#include <mlir/IR/PatternMatch.h>

using namespace mlir;

namespace dmc {

namespace {

class OpaqueTypeLower {
public:
  OpaqueTypeLower(DialectOp dialectOp, OperationOp opOp)
      : dialectName{dialectOp.getName()},
        opOp{opOp} {}

  Type tryLowerOpaqueType(Type type);
  template <typename TypeRange> Optional<std::vector<Type>>
  tryLowerOpaqueTypes(TypeRange types, StringRef name);
  LogicalResult lowerTypes();

private:
  StringRef dialectName;
  OperationOp opOp;
};

Type parseOpaqueType(OpaqueType opaqueTy) {
  std::string buf;
  llvm::raw_string_ostream os{buf};
  opaqueTy.print(os);
  return mlir::parseType(os.str(), opaqueTy.getContext());
}

Type OpaqueTypeLower::tryLowerOpaqueType(Type type) {
  auto opaqueTy = type.dyn_cast<OpaqueType>();
  if (opaqueTy && opaqueTy.getDialectNamespace() == dialectName) {
    return parseOpaqueType(opaqueTy);
  }
  return type;
}

template <typename TypeRange>
Optional<std::vector<Type>> OpaqueTypeLower::tryLowerOpaqueTypes(
    TypeRange types, StringRef name) {
  std::vector<Type> newTypes;
  newTypes.reserve(llvm::size(types));
  unsigned idx = 0;
  for (auto type : types) {
    if (auto newType = tryLowerOpaqueType(type)) {
      newTypes.push_back(newType);
    } else {
      opOp.emitOpError("failed to parse type for ") << name << " #" << idx;
      return llvm::None;
    }
    ++idx;
  }
  return newTypes;
}

LogicalResult OpaqueTypeLower::lowerTypes() {
  auto opTy = opOp.getType();
  auto argTys = tryLowerOpaqueTypes(opTy.getInputs(), "operand");
  auto resTys = tryLowerOpaqueTypes(opTy.getResults(), "result");
  if (!argTys || !resTys)
    return failure();
  auto newOpTy = mlir::FunctionType::get(*argTys, *resTys, opOp.getContext());
  if (newOpTy != opTy)
    opOp.setOpType(newOpTy);
  return success();
}

} // end anonymous namespace


LogicalResult lowerOpaqueTypes(DialectOp dialectOp) {
  for (auto opOp : dialectOp.getOps<OperationOp>()) {
    OpaqueTypeLower lower{dialectOp, opOp};
    if (failed(lower.lowerTypes()))
      return failure();
  }
  return success();
}

} // end namespace dmc
