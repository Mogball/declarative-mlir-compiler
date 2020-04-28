#include "dmc/IO/ModuleWriter.h"
#include "dmc/Dynamic/DynamicOperation.h"

using namespace mlir;

namespace dmc {

ModuleWriter::ModuleWriter(DynamicContext *ctx)
    : builder{ctx->getContext()},
      // TODO location data in Python file of call
      module{ModuleOp::create(builder.getUnknownLoc())} {}

FuncOp ModuleWriter::createFunction(
    StringRef name,
    ArrayRef<Type> argTys, ArrayRef<Type> retTys) {
  auto funcType = builder.getFunctionType(argTys, retTys);
  // TODO location data
  auto funcOp = FuncOp::create(builder.getUnknownLoc(), name, funcType);
  module.push_back(funcOp);
  return funcOp;
}

FunctionWriter::FunctionWriter(FuncOp func) 
    : builder{func.getContext()},
      func(func),
      entryBlock{func.addEntryBlock()} {
  builder.setInsertionPointToStart(entryBlock);
}

Operation *FunctionWriter::createOp(
    DynamicOperation *op, ValueRange args, ArrayRef<Type> retTys) {
  return createOp(op->getOpInfo(), args, retTys);
}

Operation *FunctionWriter::createOp(
    StringRef name, ValueRange args, ArrayRef<Type> retTys) {
  return createOp(OperationName{name, func.getContext()}, args, retTys);
}

Operation *FunctionWriter::createOp(
    OperationName opName, ValueRange args, ArrayRef<Type> retTys) {
  // TODO location data
  OperationState state{builder.getUnknownLoc(), opName};
  state.addOperands(args);
  state.addTypes(retTys);
  return builder.createOperation(state);
}

} // end namespace dmc
