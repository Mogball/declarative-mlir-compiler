#include <dmc/Dynamic/DynamicType.h>

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>

using namespace ::dmc;

namespace mlir {
namespace lua {

Optional<Type> convertLuaType(Type t) {
  auto loc = UnknownLoc::get(t.getContext());
  if (buildDynamicType("lua", "value", {}, loc) == t) {
    return getAliasedType("luallvm", "value", t.getContext());
  } else if (buildDynamicType("luac", "string", {}, loc) == t) {
    return getAliasedType("luallvm", "string", t.getContext());
  }
  return {};
}

struct ConvertLoadString : public RewritePattern {
  explicit ConvertLoadString(MLIRContext *ctx)
      : RewritePattern{"luac.load_string", 0, ctx} {}

  LogicalResult match(Operation *) const override {
    return success();
  }

  void rewrite(Operation *op, PatternRewriter &r) const override {
    using namespace LLVM;
    auto m = op->getParentOfType<ModuleOp>();
    auto *llvmDialect = op->getContext()->getRegisteredDialect<LLVMDialect>();

    auto value = op->getAttrOfType<StringAttr>("value");
    auto sz = value.getValue().size();
    std::string symName{"lua_string_"};
    symName +=
        std::to_string(reinterpret_cast<uintptr_t>(value.getAsOpaquePointer()));

    auto global = m.lookupSymbol<GlobalOp>(symName);
    if (!global) {
      auto i8Ty = LLVMType::getInt8Ty(llvmDialect);
      auto i8ArrTy = LLVMType::getArrayTy(i8Ty, sz);
      r.setInsertionPointToStart(m.getBody());
      global = r.create<GlobalOp>(op->getLoc(), i8ArrTy, false,
                                  Linkage::Internal, symName, value);
      r.setInsertionPointAfter(op);
    }

    auto addrOf = r.create<AddressOfOp>(op->getLoc(), global);
    auto i64Ty = LLVMType::getInt64Ty(llvmDialect);
    auto const0 = r.create<ConstantOp>(op->getLoc(), i64Ty,
                                       r.getI64IntegerAttr(0));

    auto i8PtrTy = LLVMType::getInt8PtrTy(llvmDialect);
    SmallVector<Value, 2> indices{const0.res(), const0.res()};
    auto gep = r.create<GEPOp>(op->getLoc(), i8PtrTy, addrOf.res(), indices);

    auto i32Ty = LLVMType::getInt32Ty(llvmDialect);
    auto constSz = r.create<ConstantOp>(op->getLoc(), i32Ty,
                                        r.getI32IntegerAttr(sz));

    SmallVector<Value, 2> vals{gep.res(), constSz.res()};
    r.replaceOp(op, vals);
  }
};

struct LuaToLLVMLoweringPass : public OperationPass<ModuleOp> {
  explicit LuaToLLVMLoweringPass()
      : OperationPass{TypeID::get<LuaToLLVMLoweringPass>()} {}

  StringRef getName() const override {
    return "convert-lua-to-llvm";
  }

  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<LuaToLLVMLoweringPass>();
  }

  void runOnOperation() override {
    LLVMTypeConverterCustomization customs;
    LLVMTypeConverter converter{&getContext(), customs};
    converter.addConversion(&convertLuaType);

    OwningRewritePatternList patterns;
    populateStdToLLVMConversionPatterns(converter, patterns,
                                        /*emitCWrappers=*/false,
                                        /*useAlignedAlloc=*/false);
    patterns.insert<ConvertLoadString>(&getContext());

    LLVMConversionTarget target{getContext()};
    if (failed(applyPartialConversion(getOperation(), target, patterns,
                                      &converter)))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createLowerLuaToLLVMPass() {
  return std::make_unique<LuaToLLVMLoweringPass>();
}

} // end namespace lua
} // end namespace mlir
