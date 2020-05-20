#include "Expose.h"

#include <boost/python.hpp>

using namespace boost::python;

BOOST_PYTHON_MODULE(mlir) {
  mlir::py::exposeParser();
}
