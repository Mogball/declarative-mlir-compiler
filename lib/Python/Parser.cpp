#include "Context.h"
#include "Utility.h"

#include <mlir/Parser.h>
#include <mlir/IR/Module.h>
#include <llvm/Support/SourceMgr.h>

using namespace llvm;

namespace mlir {
namespace py {

/// Parse a source file from a given filename. Provide a source manager and
/// a diagnostic handler for the parse.
ModuleOp parseSourceFile(std::string filename) {
  // TODO 100% a memory leak. The SourceMgr needs to be kept alive. Make python
  // manage the lifetime of the SourceMgr.
  auto *sourceMgr = new SourceMgr;
  new SourceMgrDiagnosticHandler{*sourceMgr, getMLIRContext()};
  auto ret = parseSourceFile(filename, *sourceMgr, getMLIRContext());
  return ret.release();
}

} // end namespace py
} // end namespace mlir
