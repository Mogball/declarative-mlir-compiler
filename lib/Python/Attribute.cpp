#include "Context.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/AffineMap.h>
#include <pybind11/operators.h>

using namespace pybind11;

namespace mlir {
namespace py {

bool isAffineMapAttr(Attribute attr) {
  return attr.isa<AffineMapAttr>();
}

AffineMap getAsAffineMap(Attribute attr) {
  if (!isAffineMapAttr(attr))
    throw std::invalid_argument{
        "Attribute is not an AffineMapAttr. Check with `isAffineMap`."};
  return attr.cast<AffineMapAttr>().getValue();
}

bool isArrayAttr(Attribute attr) {
  return attr.isa<ArrayAttr>();
}

Attribute getArrayAttr(const std::vector<Attribute> &value) {
  return ArrayAttr::get(value, getMLIRContext());
}

ArrayAttr asArrayAttr(Attribute attr) {
  if (!isArrayAttr(attr))
    throw std::invalid_argument{
        "Attribute is not an ArrayAttr. Check with `isArray`."};
  return attr.cast<ArrayAttr>();
}

std::vector<Attribute> *getAsArray(Attribute attr) {
  auto arr = asArrayAttr(attr);
  return new std::vector<Attribute>{std::begin(arr), std::end(arr)};
}

std::vector<Attribute> *getArraySlice(Attribute attr, slice s) {
  auto arr = asArrayAttr(attr);
  ssize_t start, stop, step, sliceLength;
  if (!s.compute(arr.size(), &start, &stop, &step, &sliceLength))
    throw error_already_set{};
  auto *ret = new std::vector<Attribute>{};
  ret->reserve(static_cast<size_t>(sliceLength));
  for (unsigned i = 0; i < sliceLength; ++i) {
    ret->push_back(arr[start]);
    start += step;
  }
  return ret;
}

} // end namespace py
} // end namespace mlir
