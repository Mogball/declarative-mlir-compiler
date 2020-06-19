#pragma once

#include <pybind11/pybind11.h>

namespace dmc {
namespace py {

pybind11::module getInternalModule();

inline auto getInternalScope() {
  return getInternalModule().attr("__dict__");
}

} // end namespace py
} // end namespace dmc
