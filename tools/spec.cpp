#include "dmc/Spec/SpecDialect.h"
#include "dmc/Traits/Registry.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Parser.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/Verifier.h>

using namespace mlir;
using namespace llvm;
using namespace dmc;

static DialectRegistration<SpecDialect> specDialectRegistration;
static DialectRegistration<TraitRegistry> registerTraits;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    llvm::errs() << "Usage: spec <mlir_in>\n";
    return -1;
  }

  MLIRContext ctx;
  SourceMgr srcMgr;
  SourceMgrDiagnosticHandler srcMgrDiagHandler{srcMgr, &ctx};
  auto mlirModule = mlir::parseSourceFile(argv[1], srcMgr, &ctx);
  if (!mlirModule) {
    llvm::errs() << "Failed to load MLIR file: " << argv[1] << "\n";
    return -1;
  }
  if (failed(verify(*mlirModule))) {
    llvm::errs() << "Failed to verify MLIR module: " << argv[1] << "\n";
    return -1;
  }
  mlirModule->print(llvm::outs());
  llvm::outs() << "\n";
}
