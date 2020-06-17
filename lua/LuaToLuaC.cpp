#include <dmc/Dynamic/DynamicType.h>

#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Pass/Pass.h>

using namespace ::dmc;

namespace mlir {
namespace lua {

class BaseWrapRewrite : public RewritePattern {
public:
  explicit BaseWrapRewrite(StringRef argTy, StringRef newOpName,
                           StringRef rootOp, MLIRContext *ctx)
      : RewritePattern{rootOp, 0, ctx},
        argTy{argTy},
        newOpName{newOpName} {}

  void rewrite(Operation *op, PatternRewriter &r) const override {
    OperationState state{op->getLoc(), newOpName};
    state.operands.push_back(*op->operand_begin());
    state.types.push_back(*op->result_type_begin());
    auto *newOp = r.createOperation(state);
    r.replaceOp(op, newOp->getResults());
  }

protected:
  StringRef argTy, newOpName;
};

class WrapRewrite : public BaseWrapRewrite {
public:
  explicit WrapRewrite(StringRef argTy, StringRef newOpName, MLIRContext *ctx)
      : BaseWrapRewrite{argTy, newOpName, "lua.wrap", ctx} {}

  LogicalResult match(Operation *op) const override {
    return success(*op->operand_type_begin() ==
                   getAliasedType("lua", argTy, op->getContext()));
  }
};

struct WrapIntegerRewrite : public WrapRewrite {
  explicit WrapIntegerRewrite(MLIRContext *ctx)
      : WrapRewrite{"integer", "luac.wrap_int", ctx} {}
};

struct WrapRealRewrite : public WrapRewrite {
  explicit WrapRealRewrite(MLIRContext *ctx)
      : WrapRewrite{"real", "luac.wrap_real", ctx} {}
};

struct WrapBoolRewrite : public WrapRewrite {
  explicit WrapBoolRewrite(MLIRContext *ctx)
      : WrapRewrite{"bool", "luac.wrap_bool", ctx} {}
};

class UnwrapRewrite : public BaseWrapRewrite {
public:
  explicit UnwrapRewrite(StringRef argTy, StringRef newOpName, MLIRContext *ctx)
      : BaseWrapRewrite{argTy, newOpName, "lua.unwrap", ctx} {}

  LogicalResult match(Operation *op) const override {
    return success(*op->result_type_begin() ==
                   getAliasedType("lua", argTy, op->getContext()));
  }
};

struct UnwrapIntegerRewrite : public UnwrapRewrite {
  explicit UnwrapIntegerRewrite(MLIRContext *ctx)
      : UnwrapRewrite{"integer", "luac.unwrap_int", ctx} {}
};

struct UnwrapRealRewrite : public UnwrapRewrite {
  explicit UnwrapRealRewrite(MLIRContext *ctx)
      : UnwrapRewrite{"real", "luac.unwrap_real", ctx} {}
};

struct UnwrapBoolRewrite : public UnwrapRewrite {
  explicit UnwrapBoolRewrite(MLIRContext *ctx)
      : UnwrapRewrite{"bool", "luac.unwrap_bool", ctx} {}
};

struct GetStringRewrite : public RewritePattern {
  explicit GetStringRewrite(MLIRContext *ctx)
      : RewritePattern{"lua.get_string", 0, ctx} {}

  LogicalResult
  matchAndRewrite(Operation *op, PatternRewriter &r) const override {
    auto value = op->getAttrOfType<StringAttr>("value");

    Operation *loadStr; {
      OperationState state{op->getLoc(), "luac.load_string"};
      state.addAttribute("value", value);
      state.addTypes({buildDynamicType("luac", "string", {}, op->getLoc()),
                      r.getIntegerType(32)});
      loadStr = r.createOperation(state);
    }

    Operation *getStr; {
      OperationState state{op->getLoc(), "luac.get_string"};
      state.addOperands(loadStr->getResults());
      state.addTypes(buildDynamicType("lua", "value", {}, op->getLoc()));
      getStr = r.createOperation(state);
    }

    r.replaceOp(op, getStr->getResults());
    return success();
  }
};

void populateLuaToLuaCConversionsPatterns(OwningRewritePatternList &patterns,
                                          MLIRContext *ctx) {
  patterns.insert<
    WrapIntegerRewrite,
    WrapRealRewrite,
    WrapBoolRewrite,
    UnwrapIntegerRewrite,
    UnwrapRealRewrite,
    UnwrapBoolRewrite,
    GetStringRewrite>(ctx);
}

struct LuaToLuaCLoweringPass : public OperationPass<ModuleOp> {
  explicit LuaToLuaCLoweringPass()
      : OperationPass{TypeID::get<LuaToLuaCLoweringPass>()} {}

  StringRef getName() const override {
    return "convert-lua-to-luac";
  }

  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<LuaToLuaCLoweringPass>();
  }

  void runOnOperation() override {
    OwningRewritePatternList patterns;
    populateLuaToLuaCConversionsPatterns(patterns, &getContext());

    ConversionTarget target{getContext()};
    target.addLegalDialect("luac");

    if (failed(applyPartialConversion(getOperation(), target, patterns,
                                      nullptr)))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createLowerLuaToLuaCPass() {
  return std::make_unique<LuaToLuaCLoweringPass>();
}

} // end namespace lua
} // end namespace mlir
