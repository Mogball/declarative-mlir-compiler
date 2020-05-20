#include "Context.h"
#include "Exception.h"

#include <mlir/IR/Identifier.h>

namespace mlir {
namespace py {

Identifier getIdentifierChecked(std::string id){
  if (id.empty())
    throw std::runtime_error{"Identifier cannot be an empty string."};
  if (id.find('\0') != std::string::npos)
    throw std::runtime_error{"Identifier cannot contain null characters."};
  return Identifier::get(id, getMLIRContext());
}

} // end namespace py
} // end namespace mlir
