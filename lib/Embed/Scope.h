#pragma once

#include <pybind11/embed.h>

inline auto getMainScope() {
  return pybind11::module::import("__main__").attr("__dict__");
}
