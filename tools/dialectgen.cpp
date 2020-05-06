#include "dmc/Spec/DialectGen.h"
#include "dmc/Spec/SpecDialect.h"
#include "dmc/Spec/DialectGen.h"
#include "dmc/Traits/Registry.h"

#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/Parser.h>
#include <mlir/Analysis/Verifier.h>

using namespace mlir;
using namespace dmc;

static DialectRegistration<SpecDialect> specDialectRegistration;
static DialectRegistration<TraitRegistry> registerTraits;

int loadModule(StringRef inFile, MLIRContext *ctx, OwningModuleRef &module) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inFile);
  if (auto ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc{});
  module = mlir::parseSourceFile(sourceMgr, ctx);
  if (!module) {
    llvm::errs() << "Failed to load file: " << inFile << "\n";
    return -1;
  }
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    llvm::errs() << "Usage: gen <dialect_mlir> <module_mlir>\n";
    return -1;
  }

  MLIRContext ctx;
  DynamicContext dynCtx{&ctx};

  OwningModuleRef dialectModule;
  StringRef dialectInFile{argv[1]};
  if (auto err = loadModule(dialectInFile, &ctx, dialectModule))
    return err;
  if (failed(verify(*dialectModule))) {
    llvm::errs() << "Failed to verify dialect module: "
        << dialectInFile << "\n";
    return -1;
  }

  registerAllDialects(*dialectModule, &dynCtx);

  OwningModuleRef mlirModule;
  StringRef mlirInFile{argv[2]};
  if (auto err = loadModule(mlirInFile, &ctx, mlirModule))
    return err;
  if (failed(verify(*mlirModule))) {
    llvm::errs() << "Failed to verify MLIR module: "
        << mlirInFile << "\n";
    return -1;
  }

  mlirModule->print(llvm::outs());
  llvm::outs() << "\n";

  return 0;
}
