#include "Context.h"

#include <mlir/IR/Identifier.h>

namespace mlir {
namespace py {

Identifier getIdentifierChecked(std::string id){
  if (id.empty())
    throw std::invalid_argument{"Identifier cannot be an empty string."};
  if (id.find('\0') != std::string::npos)
    throw std::invalid_argument{"Identifier cannot contain null characters."};
  return Identifier::get(id, getMLIRContext());
}

} // end namespace py
} // end namespace mlir
