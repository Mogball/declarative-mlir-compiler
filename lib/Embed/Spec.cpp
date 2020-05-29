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
  return Base::getChecked(loc, Kind, expr);
}

LogicalResult PyType::verifyConstructionInvariants(Location loc,
                                                   StringRef expr) {
  return py::verifyTypeConstraint(loc, expr);
}

LogicalResult PyType::verify(Type ty) {
  return py::evalConstraintExpr(getImpl()->expr, ty);
}

void PyType::print(DialectAsmPrinter &printer) {
  printer << getTypeName() << "<\"" << getImpl()->expr << "\">";
}

/// PyAttr implementation.
PyAttr PyAttr::getChecked(Location loc, StringRef expr) {
  return Base::getChecked(loc, Kind, expr);
}

LogicalResult PyAttr::verifyConstructionInvariants(Location loc,
                                                   StringRef expr) {
  return py::verifyAttrConstraint(loc, expr);
}

LogicalResult PyAttr::verify(Attribute attr) {
  return py::evalConstraintExpr(getImpl()->expr, attr);
}

void PyAttr::print(DialectAsmPrinter &printer) {
  printer << getAttrName() << "<\"" << getImpl()->expr << "\">";
}

} // end namespace dmc
