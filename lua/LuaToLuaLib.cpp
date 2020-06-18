#include <dmc/Dynamic/DynamicType.h>

#include <mlir/IR/PatternMatch.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Pass/Pass.h>

using namespace ::dmc;

namespace mlir {
namespace lua {

class LowerToFuncCall : public RewritePattern {
public:
  explicit LowerToFuncCall(StringRef rootOp, StringRef funcName,
                           MLIRContext *ctx)
      : RewritePattern{rootOp, 0, ctx},
        funcName{funcName} {}

  LogicalResult match(Operation *) const override {
    return success();
  }

  void rewrite(Operation *op, PatternRewriter &r) const override {
    auto valTy = buildDynamicType("lua", "value", {}, op->getLoc());
    if (op->getNumResults() == 1 && valTy == *op->result_type_begin()) {
      OperationState state{op->getLoc(), "lualib.alloc"};
      state.addTypes(valTy);
      auto *alloc = r.createOperation(state);

      SmallVector<Type, 0> tys;
      SmallVector<Value, 4> args;
      args.push_back(alloc->getResult(0));
      args.append(op->operand_begin(), op->operand_end());
      r.create<CallOp>(op->getLoc(), funcName, tys, args);

      r.replaceOp(op, alloc->getResults());
    } else {
      auto call = r.create<CallOp>(op->getLoc(), funcName, op->getResultTypes(),
                                   op->getOperands());
      r.replaceOp(op, call.getResults());
    }
  }

private:
  StringRef funcName;
};

void populateLuaToLuaLibConversionPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *ctx) {
#define LOWER_TO_FUNC(dialect, rootOp, funcName) \
  struct LowerToFunc_##rootOp : public LowerToFuncCall { \
    LowerToFunc_##rootOp(MLIRContext *ctx) \
        : LowerToFuncCall{#dialect "." #rootOp, #funcName, ctx} {} \
  }; \
  patterns.insert<LowerToFunc_##rootOp>(ctx)

#define LUA_TO_FUNC(rootOp, funcName) LOWER_TO_FUNC(lua, rootOp, funcName)
#define LUAC_TO_FUNC(rootOp, funcName) LOWER_TO_FUNC(luac, rootOp, funcName)

  LUA_TO_FUNC(add, lua_add);
  LUA_TO_FUNC(sub, lua_sub);
  LUA_TO_FUNC(eq, lua_eq);
  LUA_TO_FUNC(neq, lua_neq);
  LUA_TO_FUNC(get_nil, lua_get_nil);
  LUA_TO_FUNC(new_table, lua_new_table);
  LUAC_TO_FUNC(get_string, lua_get_string);
  LUAC_TO_FUNC(wrap_int, lua_wrap_int);
  LUAC_TO_FUNC(wrap_real, lua_wrap_real);
  LUAC_TO_FUNC(wrap_bool, lua_wrap_bool);
  LUAC_TO_FUNC(unwrap_int, lua_unwrap_int);
  LUAC_TO_FUNC(unwrap_real, lua_unwrap_real);
  LUAC_TO_FUNC(unwrap_bool, lua_unwrap_bool);
  LUA_TO_FUNC(typeof, lua_typeof);
  LUA_TO_FUNC(table_get, lua_table_get);
  LUA_TO_FUNC(table_set, lua_table_set);
  LUA_TO_FUNC(table_size, lua_table_size);
}

struct LuaToLuaLibLoweringPass : public OperationPass<ModuleOp> {
  explicit LuaToLuaLibLoweringPass()
      : OperationPass{TypeID::get<LuaToLuaLibLoweringPass>()} {}

  StringRef getName() const override {
    return "convert-lua-to-lua-lib";
  }

  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<LuaToLuaLibLoweringPass>();
  }

  void runOnOperation() override {
    OwningRewritePatternList patterns;
    populateLuaToLuaLibConversionPatterns(patterns, &getContext());

    ConversionTarget target{getContext()};
    target.addLegalDialect("std");
    target.addLegalDialect("lualib");

    if (failed(applyPartialConversion(getOperation(), target, patterns,
                                      nullptr)))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createLowerLuaToLuaLibPass() {
  return std::make_unique<LuaToLuaLibLoweringPass>();
}

} // end namespace lua
} // end namespace mlir
