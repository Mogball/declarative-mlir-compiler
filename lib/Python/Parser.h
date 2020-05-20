#include <mlir/IR/Module.h>

namespace mlir {
namespace py {

std::unique_ptr<OwningModuleRef> parseSourceFile(std::string filename);

} // end namespace py
} // end namespace mlir
