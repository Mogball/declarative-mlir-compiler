#include "dmc/Spec/SpecDialect.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Parser.h>
#include <mlir/IR/Module.h>

using namespace mlir;
using namespace dmc;

static DialectRegistration<SpecDialect> specDialectRegistration;

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
  if (argc != 2) {
    llvm::errs() << "Usage: spec <mlir_in>\n";
    return -1;
  }

  MLIRContext ctx;
  OwningModuleRef module;
  StringRef inFile{argv[1]};
  if (auto err = loadModule(inFile, &ctx, module))
    return err;
  module->print(llvm::outs());
  llvm::outs() << "\n";
}
