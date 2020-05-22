#include "Context.h"
#include "Identifier.h"
#include "Attribute.h"
#include "Location.h"

#include <mlir/IR/AffineMap.h>
#include <mlir/IR/OperationSupport.h>

using namespace pybind11;

namespace mlir {
namespace py {

ArrayAttr getArrayAttr(const std::vector<Attribute> &attrs) {
  return ArrayAttr::get(attrs, getMLIRContext());
}

AttributeArray *getArrayAttrValue(ArrayAttr attr) {
  return new AttributeArray{attr.getValue()};
}

auto wrapIndex(ptrdiff_t i, unsigned sz) {
  if (i < 0)
    i += sz;
  if (i < 0 || static_cast<unsigned>(i) >= sz)
    throw index_error{};
  return i;
}

AttributeArray *arrayGetSlice(ArrayAttr attr, pybind11::slice s) {
  size_t start, stop, step, sliceLength;
  if (!s.compute(attr.size(), &start, &stop, &step, &sliceLength))
    throw error_already_set{};
  auto *ret = new AttributeArray;
  ret->reserve(sliceLength);
  for (unsigned i = 0; i < sliceLength; ++i) {
    ret->push_back(attr[start]);
    start += step;
  }
  return ret;
}

Attribute arrayGetIndex(ArrayAttr attr, ptrdiff_t i) {
  i = wrapIndex(i, attr.size());
  return attr[static_cast<unsigned>(i)];
}

DictionaryAttr getDictionaryAttr(const AttributeMap &attrs) {
  NamedAttrList attrList;
  for (auto &[name, attr] : attrs) {
    attrList.push_back({getIdentifierChecked(name), attr});
  }
  return DictionaryAttr::get(attrList, getMLIRContext());
}

AttributeMap *getDictionaryAttrValue(DictionaryAttr attr) {
  auto *ret = new AttributeMap;
  ret->reserve(attr.size());
  for (auto &[name, attr] : attr) {
    ret->emplace(name.strref(), attr);
  }
  return ret;
}

Attribute dictionaryAttrGetItem(DictionaryAttr attr, const std::string &key) {
  auto ret = attr.get(key);
  if (!ret)
    throw key_error{};
  return ret;
}

FloatAttr getFloatAttr(Type ty, double val, Location loc) {
  if (!ty)
    throw std::invalid_argument{"Float type cannot be null"};
  if (failed(FloatAttr::verifyConstructionInvariants(
        loc, ty, val)))
    throw std::invalid_argument{"Bad float representation"};
  return FloatAttr::get(ty, val);
}

IntegerAttr getIntegerAttr(Type ty, int64_t val, Location loc) {
  if (!ty)
    throw std::invalid_argument{"Integer type cannot be null"};
  if (failed(IntegerAttr::verifyConstructionInvariants(
        loc, ty, val)))
    throw std::invalid_argument{"Bad integer representation"};
  return IntegerAttr::get(ty, val);
}

OpaqueAttr getOpaqueAttr(const std::string &dialect, const std::string &data,
                         Type type, Location loc) {
  auto id = getIdentifierChecked(dialect);
  if (failed(OpaqueAttr::verifyConstructionInvariants(loc, id, data, type)))
    throw std::invalid_argument{"Invalid OpaqueAttr construction"};
  return OpaqueAttr::get(id, data, type, getMLIRContext());
}

std::string getOpaqueAttrDialect(OpaqueAttr attr) {
  return attr.getDialectNamespace().str();
}

std::string getOpaqueAttrData(OpaqueAttr attr) {
  return attr.getAttrData().str();
}

} // end namespace py
} // end namespace mlir
