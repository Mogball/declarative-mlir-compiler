#pragma once

#include <mlir/IR/Identifier.h>

namespace mlir {
namespace py {

Identifier getIdentifierChecked(std::string id);

} // end namespace py
} // end namespace mlir
