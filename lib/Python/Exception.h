#pragma once

#include <exception>

namespace mlir {
namespace py {

class invalid_cast : public std::runtime_error {
public:
  invalid_cast(std::string msg) : runtime_error{msg} {}
};

} // end namespace py
} // end namespace mlir
