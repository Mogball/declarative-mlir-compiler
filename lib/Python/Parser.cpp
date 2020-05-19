#include "dmc/Python/Expose.h"

#include <mlir/Parser.h>
#include <mlir/IR/Module.h>
#include <boost/python.hpp>

using namespace mlir;

namespace mlir {
namespace py {

/// OwningModuleRef is a smart pointer to ModuleOp and does not support copy
/// construction or assignment. Allocate it to a pointer and tell Python to
/// manage its lifespan.
OwningModuleRef *moveToHeap(OwningModuleRef &&moduleRef) {
  auto *ret = new OwningModuleRef;
  *ret = std::move(moduleRef);
  return ret;
}

// OwningModuleRef parseSourceFile(StringRef, MLIRContext *)
OwningModuleRef *parseSourceFile(std::string filename) {
  auto ret = parseSourceFile(filename, getMLIRContext());
  if (ret)
    return moveToHeap(std::move(ret));
  return nullptr; // return None if failed
}

void exposeParser() {
  using namespace boost;
  using namespace boost::python;
  class_<OwningModuleRef, noncopyable>("OwningModuleRef", no_init);
  def("parseSourceFile", parseSourceFile,
      return_value_policy<manage_new_object>{});
}

} // end namespace py
} // end namespace mlir
