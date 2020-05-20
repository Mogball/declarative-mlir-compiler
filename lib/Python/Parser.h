#include <mlir/IR/Module.h>

namespace mlir {
namespace py {

OwningModuleRef *parseSourceFile(std::string filename);

} // end namespace py
} // end namespace mlir
