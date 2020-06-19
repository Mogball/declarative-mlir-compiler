#pragma once

#include <pybind11/pybind11.h>

namespace dmc {
class DynamicDialect;
namespace py {
pybind11::module exposeDialect(DynamicDialect *dialect);
} // end namespace py
} // end namespace dmc
