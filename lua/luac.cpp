#include <dmc/Spec/SpecDialect.h>
#include <dmc/Spec/DialectGen.h>
#include <dmc/Traits/Registry.h>

#include <mlir/InitAllDialects.h>
#include <mlir/Parser.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h>

#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/ToolOutputFile.h>

using namespace llvm;
using namespace mlir;
using namespace dmc;

static DialectRegistration<SpecDialect> specDialectRegistration;
static DialectRegistration<TraitRegistry> registerTraits;
static DialectRegistration<StandardOpsDialect> registerStdOps;
static DialectRegistration<scf::SCFDialect> registerScfOps;
static DialectRegistration<LLVM::LLVMDialect> registerLlvmOps;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"));

static cl::opt<std::string> dialectFilename("dialect",
                                            cl::desc("Dialect filename"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::init("-"));

static cl::opt<std::string> libFilename("lualib",
                                        cl::desc("Library functions"));

static cl::opt<bool> lowerToLuac("lower-to-luac", cl::desc("Lower to LuaC"),
                                 cl::init(false));
static cl::opt<bool> lowerToLuaLib("lower-to-lua-lib",
                                   cl::desc("Lower to Lua Lib"),
                                   cl::init(false));
static cl::opt<bool> loopToStd("loop-to-std", cl::desc("Lower SCF to STD"),
                               cl::init(false));
static cl::opt<bool> lowerToLLVM("lower-to-llvm", cl::desc("Lower all to LLVM"),
                                 cl::init(false));

static cl::opt<bool> lowerAll("lower-all", cl::desc("Lower all the way to LLVM"),
                              cl::init(false));

namespace mlir {
namespace lua {
extern std::unique_ptr<Pass> createLowerLuaToLuaCPass();
extern std::unique_ptr<Pass> createLowerLuaToLuaLibPass();
extern std::unique_ptr<Pass> createLowerLuaToLLVMPass();
} // end namespace lua
} // end namespace mlir

OwningModuleRef loadAndVerify(StringRef filename, SourceMgr &mgr,
                              MLIRContext *ctx) {
  auto moduleRef = mlir::parseSourceFile(filename, mgr, ctx);
  if (!moduleRef) {
    errs() << "Failed to load/parse module: " << filename << "\n";
    return {};
  }
  if (failed(verify(*moduleRef))) {
    errs() << "Failed to verify module: " << filename << "\n";
    return {};
  }
  return moduleRef;
}

LogicalResult appendLibDecls(OwningModuleRef &moduleRef) {
  SourceMgr mgr;
  SourceMgrDiagnosticHandler diag{mgr, moduleRef->getContext()};
  auto declModule = loadAndVerify(libFilename, mgr, moduleRef->getContext());
  if (!declModule)
    return failure();
  for (auto func : declModule->getOps<FuncOp>()) {
    moduleRef->push_back(func.clone());
  }
  return success();
}

int main(int argc, char *argv[]) {
  cl::ParseCommandLineOptions(argc, argv);
  llvm::InitLLVM y{argc, argv};

  MLIRContext ctx;
  auto *dynCtx = new DynamicContext{&ctx};

  SourceMgr dialectMgr;
  SourceMgrDiagnosticHandler dialectDiag{dialectMgr, &ctx};
  if (auto dialectModule = loadAndVerify(dialectFilename, dialectMgr, &ctx)) {
    if (failed(registerAllDialects(*dialectModule, dynCtx))) {
      errs() << "Failed to register dynamic dialects\n";
      return -1;
    }
  } else {
    return -1;
  }

  SourceMgr mlirMgr;
  SourceMgrDiagnosticHandler mlirDiag{mlirMgr, &ctx};
  auto mlirModule = loadAndVerify(inputFilename, mlirMgr, &ctx);
  if (!mlirModule) {
    return -1;
  }
  if (failed(verify(*mlirModule))) {
    errs() << "Failed to verify MLIR module\n";
    return -1;
  }

  PassManager pm{&ctx};
  if (lowerAll || lowerToLuac) {
    pm.addPass(mlir::lua::createLowerLuaToLuaCPass());
    if (lowerAll || lowerToLuaLib) {
      if (failed(appendLibDecls(mlirModule))) {
        return -1;
      }
      pm.addPass(mlir::lua::createLowerLuaToLuaLibPass());
    }
  }
  if (lowerAll || loopToStd) {
    pm.addPass(mlir::createLowerToCFGPass());
  }
  if (lowerAll || lowerToLLVM) {
    pm.addPass(mlir::lua::createLowerLuaToLLVMPass());
  }

  if (failed(pm.run(*mlirModule))) {
    errs() << "Pass manager failed\n";
    return -1;
  }

  std::string errorMessage;
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    errs() << errorMessage << "\n";
    return -1;
  }

  mlirModule->print(output->os());
  output->os() << "\n";
  output->keep();
  return 0;
}
