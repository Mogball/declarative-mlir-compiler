#include "dmc/Embed/Constraints.h"
#include "dmc/Spec/SpecTypes.h"
#include "dmc/Spec/SpecAttrs.h"

using namespace mlir;

namespace dmc {

namespace detail {

/// Store the constraint expression.
struct PyConstraintStorage {
  using KeyTy = StringRef;

  explicit PyConstraintStorage(KeyTy key) : expr{key} {}
  bool operator==(KeyTy key) const { return key == expr; }
  static llvm::hash_code hashKey(KeyTy key) { return hash_value(key); }

  StringRef expr;
  /// Not part of the key, but store the function name. Initialize to empty.
  std::string funcName{};
};

struct PyTypeStorage : public PyConstraintStorage, public TypeStorage {
  using PyConstraintStorage::PyConstraintStorage;

  static PyTypeStorage *construct(TypeStorageAllocator &alloc, KeyTy key) {
    auto expr = alloc.copyInto(key);
    return new (alloc.allocate<PyTypeStorage>()) PyTypeStorage{expr};
  }
};

struct PyAttrStorage : public PyConstraintStorage, public AttributeStorage {
  using PyConstraintStorage::PyConstraintStorage;

  static PyAttrStorage *construct(AttributeStorageAllocator &alloc, KeyTy key) {
    auto expr = alloc.copyInto(key);
    return new (alloc.allocate<PyAttrStorage>()) PyAttrStorage{expr};
  }
};

} // end namespace detail

/// PyType implementation.
PyType PyType::getChecked(Location loc, StringRef expr) {
  auto ret = Base::get(loc.getContext(), Kind, expr);
  auto &funcName = ret.getImpl()->funcName;
  if (funcName.empty()) {
    if (failed(py::registerConstraint(loc, expr, funcName)))
      return {};
  }
  return ret;
}

LogicalResult PyType::verify(Type ty) {
  return py::evalConstraint(getImpl()->funcName, ty);
}

void PyType::print(DialectAsmPrinter &printer) {
  printer << getTypeName() << "<\"" << getImpl()->expr << "\">";
}

/// PyAttr implementation.
PyAttr PyAttr::getChecked(Location loc, StringRef expr) {
  auto ret = Base::get(loc.getContext(), Kind, expr);
  auto &funcName = ret.getImpl()->funcName;
  if (funcName.empty()) {
    if (failed(py::registerConstraint(loc, expr, funcName)))
      return {};
  }
  return ret;
}

LogicalResult PyAttr::verify(Attribute attr) {
  return py::evalConstraint(getImpl()->funcName, attr);
}

void PyAttr::print(DialectAsmPrinter &printer) {
  printer << getAttrName() << "<\"" << getImpl()->expr << "\">";
}

} // end namespace dmc
