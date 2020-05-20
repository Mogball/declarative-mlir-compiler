#include "Expose.h"
#include "Exception.h"

#include <boost/python.hpp>

using namespace boost::python;
using namespace mlir::py;

void translateInvalidCast(const invalid_cast &e) {
  PyErr_SetString(PyExc_UserWarning, e.what());
}

BOOST_PYTHON_MODULE(mlir) {
  register_exception_translator<invalid_cast>(translateInvalidCast);
  exposeParser();
  exposeModule();
  exposeLocation();
}
