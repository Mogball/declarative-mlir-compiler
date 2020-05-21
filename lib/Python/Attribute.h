#include <mlir/IR/Attributes.h>
#include <pybind11/operators.h>
#include <unordered_map>

namespace mlir {
namespace py {

/// ArrayAttr.
using AttributeArray = std::vector<Attribute>;

ArrayAttr getArrayAttr(const std::vector<Attribute> &attrs);
AttributeArray *getArrayAttrValue(ArrayAttr attr);
AttributeArray *arrayGetSlice(ArrayAttr attr, pybind11::slice s);
Attribute arrayGetIndex(ArrayAttr attr, ptrdiff_t i);

/// DictionaryAttr.
using AttributeMap = std::unordered_map<std::string, Attribute>;

DictionaryAttr getDictionaryAttr(const AttributeMap &attrs);
AttributeMap *getDictionaryAttrValue(DictionaryAttr attr);
Attribute dictionaryAttrGetItem(DictionaryAttr attr, const std::string &key);

/// FloatAttr.
FloatAttr getFloatAttr(Type ty, double val);
FloatAttr getFloatAttr(Type ty, double val, Location loc);

/// IntegerAttr.
IntegerAttr getIntegerAttr(Type ty, int64_t val);
IntegerAttr getIntegerAttr(Type ty, int64_t val, Location loc);

/// OpaqueAttr.
OpaqueAttr getOpaqueAttr(const std::string &dialect, const std::string &data,
                         Type type);
OpaqueAttr getOpaqueAttr(const std::string &dialect, const std::string &data,
                         Type type, Location loc);
std::string getOpaqueAttrDialect(OpaqueAttr attr);
std::string getOpaqueAttrData(OpaqueAttr attr);

} // end namespace py
} // end namespace mlir
