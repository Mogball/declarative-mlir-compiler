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
///
/// OwningModuleRef is a smart pointer to ModuleOp and does not support copy
/// construction or assignment. Allocate it to a pointer and tell Python to
/// manage its lifespan.
std::unique_ptr<OwningModuleRef> parseSourceFile(std::string filename) {
  SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler diagnosticHandler{sourceMgr, getMLIRContext()};
  auto ret = parseSourceFile(filename, sourceMgr, getMLIRContext());
  if (ret)
    return moveToHeap(std::move(ret));
  return nullptr;
}

} // end namespace py
} // end namespace mlir
