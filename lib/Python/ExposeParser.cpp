#include "Parser.h"

#include <boost/python.hpp>

namespace mlir {
namespace py {

void exposeParser() {
  using namespace boost;
  using namespace boost::python;
  def("parseSourceFile", parseSourceFile,
      return_value_policy<manage_new_object>{});
}

} // end namespace py
} // end namespace mlir
