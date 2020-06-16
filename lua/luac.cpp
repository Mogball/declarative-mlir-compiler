#include <dmc/Spec/SpecDialect.h>
#include <dmc/Spec/DialectGen.h>
#include <dmc/Traits/Registry.h>

#include <mlir/InitAllDialects.h>
#include <mlir/Parser.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>

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

static cl::opt<bool> lowerToLuac("lower-to-luac", cl::desc("Lower to LuaC"),
                                 cl::init(false));

namespace mlir {
namespace lua {
extern std::unique_ptr<Pass> createLowerLuaToLuaCPass();
} // end namespace lua
} // end namespace mlir

int main(int argc, char *argv[]) {
  cl::ParseCommandLineOptions(argc, argv);
  llvm::InitLLVM y{argc, argv};

  MLIRContext ctx;
  auto *dynCtx = new DynamicContext{&ctx};

  SourceMgr dialectSrcMgr;
  SourceMgrDiagnosticHandler dialectDiag{dialectSrcMgr, &ctx};
  auto dialectModule = mlir::parseSourceFile(dialectFilename, dialectSrcMgr,
                                             &ctx);
  if (!dialectModule) {
    errs() << "Failed to load dialect module\n";
    return -1;
  }
  if (failed(verify(*dialectModule))) {
    errs() << "Failed to verify dialect module\n";
    return -1;
  }
  if (failed(registerAllDialects(*dialectModule, dynCtx))) {
    errs() << "Failed to register dynamic dialects\n";
    return -1;
  }

  SourceMgr mlirSrcMgr;
  SourceMgrDiagnosticHandler mlirDiag{mlirSrcMgr, &ctx};
  auto mlirModule = mlir::parseSourceFile(inputFilename, &ctx);
  if (!mlirModule) {
    errs() << "Failed to load MLIR module\n";
    return -1;
  }
  if (failed(verify(*mlirModule))) {
    errs() << "Failed to verify MLIR module\n";
    return -1;
  }

  PassManager pm{&ctx};
  if (lowerToLuac)
    pm.addPass(mlir::lua::createLowerLuaToLuaCPass());

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
