#include "dmc/Spec/SpecOps.h"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Parser.h>
#include <mlir/IR/PatternMatch.h>

using namespace mlir;

namespace dmc {

struct LowerOpaqueType : public OpRewritePattern<OperationOp> {
  LowerOpaqueType(MLIRContext *ctx)
      : OpRewritePattern<OperationOp>(ctx) {}

  LogicalResult matchAndRewrite(OperationOp opOp,
                                PatternRewriter &rewriter) const override {
    /// After custom types have been registered, parse OpaqueTypes.
    auto dialectName = dyn_cast<DialectOp>(opOp.getParentOp()).getName();
    auto opTy = opOp.getType();
    // Inputs
    auto inputTys = opTy.getInputs();
    std::vector<Type> newInputTys;
    newInputTys.reserve(llvm::size(inputTys));
    for (auto inputTy : inputTys) {
      if (auto opaqueTy = inputTy.dyn_cast<OpaqueType>()) {
        if (opaqueTy.getDialectNamespace() == dialectName) {
          std::string buf;
          llvm::raw_string_ostream sstream{buf};
          opaqueTy.print(sstream);
          auto newTy = mlir::parseType(sstream.str(), opOp.getContext());
          if (!newTy)
            return rewriter.notifyMatchFailure(
                opOp, "Failed to convert opaque type");
          newInputTys.push_back(newTy);
          continue;
        }
      }
      newInputTys.push_back(inputTy);
    }
    // Outputs
    auto resultTys = opTy.getResults();
    std::vector<Type> newResultTys;
    newResultTys.reserve(llvm::size(resultTys));
    for (auto resultTy : resultTys) {
      if (auto opaqueTy = resultTy.dyn_cast<OpaqueType>()) {
        if (opaqueTy.getDialectNamespace() == dialectName) {
          std::string buf;
          llvm::raw_string_ostream sstream{buf};
          opaqueTy.print(sstream);
          auto newTy = mlir::parseType(sstream.str(), opOp.getContext());
          if (!newTy)
            return rewriter.notifyMatchFailure(
                opOp, "Failed to convert opaque type");
          newResultTys.push_back(newTy);
          continue;
        }
      }
      newResultTys.push_back(resultTy);
    }
    /// Replace the Op type.
    auto newOpTy = mlir::FunctionType::get(newInputTys, newResultTys,
                                           opOp.getContext());

    if (newOpTy == opTy)
      return failure();
    opOp.setOpType(newOpTy);
    return success();
  }
};

void lowerOpaqueTypes(DialectOp dialectOp) {
  OwningRewritePatternList patterns;
  patterns.insert<LowerOpaqueType>(dialectOp.getContext());
  applyPatternsAndFoldGreedily(dialectOp, patterns);
}

} // end namespace dmc
