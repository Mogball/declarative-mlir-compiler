#include <mlir/IR/Attributes.h>
#include <pybind11/operators.h>

namespace mlir {
namespace py {

/// AffineMapAttr.
bool isAffineMapAttr(Attribute attr);
AffineMap getAsAffineMap(Attribute attr);

/// ArrayAttr.
bool isArrayAttr(Attribute attr);
Attribute getArrayAttr(const std::vector<Attribute> &value);
std::vector<Attribute> *getAsArray(Attribute attr);
std::vector<Attribute> *getArraySlice(Attribute attr, pybind11::slice s);
void setArrayIndex(Attribute attr, ptrdiff_t i);

} // end namespace py
} // end namespace mlir
