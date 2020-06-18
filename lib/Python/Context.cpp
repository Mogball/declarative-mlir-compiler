#include "dmc/Spec/SpecDialect.h"
#include "dmc/Traits/Registry.h"
#include "dmc/Dynamic/DynamicContext.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>

namespace mlir {
namespace py {

/// Wrap the global MLIR context in a singleton class, using member objects
/// to ensure initialization order of dialect registrations. Static objects
/// are not guaranteed by the standard to be initialized in any order.
class GlobalContextHandle {
public:
  static GlobalContextHandle &instance() {
    static GlobalContextHandle instance;
    return instance;
  }

  MLIRContext *getContext() { return ptr; }
  void setContext(MLIRContext *ctx) { ptr = ctx; }

private:
  /// Initiazation order is guaranteed.
  DialectRegistration<StandardOpsDialect> standardOpsDialect;
  DialectRegistration<scf::SCFDialect> scfDialect;
  DialectRegistration<LLVM::LLVMDialect> llvmDialect;
  DialectRegistration<dmc::SpecDialect> specDialect;
  DialectRegistration<dmc::TraitRegistry> traitRegistry;
  MLIRContext context;
  MLIRContext *ptr{&context};
};

MLIRContext *getMLIRContext() {
  return GlobalContextHandle::instance().getContext();
}

void setMLIRContext(MLIRContext *ctx) {
  GlobalContextHandle::instance().setContext(ctx);
}

} // end namespace py
} // end namespace mlir
