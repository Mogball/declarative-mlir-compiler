#include "Support.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/StandardTypes.h>

using namespace pybind11;

namespace mlir {
namespace py {

template <typename FcnT> auto nullcheck(FcnT fcn) {
  return ::nullcheck(fcn, "elements attribute");
}

bool elementsAttrHasIndex(ElementsAttr attr,
                          const std::vector<uint64_t> &index) {
  return attr.isValidIndex(index);
}

Attribute getElementsAttrValue(ElementsAttr attr,
                               const std::vector<uint64_t> &index) {
  if (!elementsAttrHasIndex(attr, index))
    throw index_error{};
  return attr.getValue(index);
}

template <typename T>
void denseElsCtorFrom(class_<DenseElementsAttr> &cls) {
  cls.def(init([](ShapedType ty, const std::vector<T> &vals) {
    return DenseElementsAttr::get(ty, llvm::makeArrayRef(vals));
  }));
  cls.def(init([](ShapedType ty, T val) {
    return DenseElementsAttr::get(ty, llvm::makeArrayRef(val));
  }));
}

void exposeElementsAttr(module &m, class_<Attribute> &attr) {
  class_<ElementsAttr> elsAttr{m, "ElementsAttr", attr};
  elsAttr
      .def_property_readonly("type", nullcheck(&ElementsAttr::getType))
      .def("getValue", nullcheck(&getElementsAttrValue))
      .def("__getitem__", nullcheck(&getElementsAttrValue))
      .def("__len__", nullcheck([](ElementsAttr attr)
                                { return attr.getNumElements(); }))
      .def("__contains__", nullcheck(&elementsAttrHasIndex));
  /// TODO mapValues once Python can be called from C++.

  class_<DenseElementsAttr> denseElsAttr{m, "DenseElementsAttr", elsAttr};
  denseElsAttr
      .def(init([](ShapedType ty,
                   const std::vector<std::string> &vals) {
        std::vector<StringRef> refs{std::begin(vals), std::end(vals)};
        return DenseElementsAttr::get(ty, refs);
      }));
  // vector<bool> does not provide data()
  denseElsCtorFrom<Attribute>(denseElsAttr);

  // Numeric types
  denseElsCtorFrom<int64_t>(denseElsAttr);
  denseElsCtorFrom<uint64_t>(denseElsAttr);
  denseElsCtorFrom<double>(denseElsAttr);
  denseElsCtorFrom<std::complex<int64_t>>(denseElsAttr);
  denseElsCtorFrom<std::complex<uint64_t>>(denseElsAttr);
  denseElsCtorFrom<std::complex<double>>(denseElsAttr);
}

} // end namespace py
} // end namespace mlir
