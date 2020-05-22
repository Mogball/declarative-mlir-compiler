#include "Support.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/StandardTypes.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

using namespace pybind11;
using namespace mlir;

namespace mlir {
namespace py {

template <typename FcnT> auto nullcheck(FcnT fcn) { return ::nullcheck(fcn, "elements attribute");
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

void checkType(ShapedType ty) {
  if (!ty)
    throw std::invalid_argument{"Shape type cannot be null"};
}

template <typename T, typename Base>
void denseElsCtorFrom(class_<Base> &cls) {
  cls.def(init([](ShapedType ty, const std::vector<T> &vals) {
    checkType(ty);
    return Base::get(ty, llvm::makeArrayRef(vals));
  }));
  cls.def(init([](ShapedType ty, T val) {
    checkType(ty);
    return Base::get(ty, llvm::makeArrayRef(val));
  }));
}

template <typename Base>
Base getDenseStringEls(ShapedType ty, const std::vector<std::string> &vals) {
  /// Convert to StringRef.
  std::vector<StringRef> refs{std::begin(vals), std::end(vals)};
  return Base::get(ty, refs);
}

Attribute getDenseElementsSplatValue(DenseElementsAttr attr) {
  if (!attr.isSplat())
    throw std::invalid_argument{"DenseElementsAttr is not a splat."};
  return attr.getSplatValue();
}

template <typename Base>
void defBasicIter(class_<Base> &cls) {
  cls.def("__iter__", nullcheck([](Base attr) {
    return make_iterator(attr.begin(), attr.end());
  }), keep_alive<0, 1>());
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
      .def(init(&getDenseStringEls<DenseElementsAttr>))
      .def(init([](ShapedType ty,
                   const std::vector<bool> &vals) {
        /// std::vector<bool> cannot be directly converted to ArrayRef<bool>.
        checkType(ty);
        SmallVector<bool, 0> boolVals{std::begin(vals), std::end(vals)};
        return DenseElementsAttr::get(ty, boolVals);
      }));
  denseElsCtorFrom<Attribute>(denseElsAttr);

  // Numeric types
  denseElsCtorFrom<int64_t>(denseElsAttr);
  denseElsCtorFrom<uint64_t>(denseElsAttr);
  denseElsCtorFrom<double>(denseElsAttr);
  denseElsCtorFrom<std::complex<int64_t>>(denseElsAttr);
  denseElsCtorFrom<std::complex<uint64_t>>(denseElsAttr);
  denseElsCtorFrom<std::complex<double>>(denseElsAttr);

  denseElsAttr
      .def("isSplat", nullcheck(&DenseElementsAttr::isSplat))
      .def("getSplatValue", nullcheck(&getDenseElementsSplatValue))
      .def("getValues", nullcheck([](DenseElementsAttr attr) {
        return make_iterator(attr.attr_value_begin(), attr.attr_value_end());
      }), keep_alive<0, 1>())
      .def("__iter__", nullcheck([](DenseElementsAttr attr) {
        return make_iterator(attr.attr_value_begin(), attr.attr_value_end());
      }), keep_alive<0, 1>())
      .def("getBoolValues", nullcheck([](DenseElementsAttr attr) {
        auto eltType = attr.getType().getElementType().dyn_cast<IntegerType>();
        if (!eltType || eltType.getWidth() != 1)
          throw std::invalid_argument{"Element type must be i1 integer type"};
        auto range = attr.getBoolValues();
        return make_iterator(range.begin(), range.end());
      }), keep_alive<0, 1>())
      .def("getIntValues", nullcheck([](DenseElementsAttr attr) {
        auto eltType = attr.getType().getElementType();
        if (!eltType.isIntOrIndex())
          throw std::invalid_argument{"Element type must be an integer type"};
        return make_iterator(attr.int_value_begin(), attr.int_value_end());
      }), keep_alive<0, 1>())
      .def("getComplexIntValues", nullcheck([](DenseElementsAttr attr) {
        auto eltType = attr.getType().getElementType().dyn_cast<ComplexType>();
        if (!eltType || !eltType.getElementType().isa<IntegerType>())
          throw std::invalid_argument{"Expected complex integral element type"};
        auto range = attr.getComplexIntValues();
        return make_iterator(range.begin(), range.end());
      }), keep_alive<0, 1>())
      .def("getFloatValues", nullcheck([](DenseElementsAttr attr) {
        auto eltType = attr.getType().getElementType();
        if (!eltType.isa<FloatType>())
          throw std::invalid_argument{"Expected floating point element type"};
        return make_iterator(attr.float_value_begin(), attr.float_value_end());
      }), keep_alive<0, 1>())
      .def("getComplexFloatValues", nullcheck([](DenseElementsAttr attr) {
        auto eltType = attr.getType().getElementType().dyn_cast<ComplexType>();
        if (!eltType || !eltType.getElementType().isa<FloatType>())
          throw std::invalid_argument{
              "Expected complex floating point element type"};
        auto range = attr.getComplexFloatValues();
        return make_iterator(range.begin(), range.end());
      }))
      .def("reshape", nullcheck([](DenseElementsAttr attr, ShapedType ty) {
        if (ty.getElementType() != attr.getType().getElementType())
          throw std::invalid_argument{"Expected the same element type"};
        if (ty.getNumElements() != attr.getType().getNumElements())
          throw std::invalid_argument{"Expected the same number of elements"};
        return attr.reshape(ty);
      }));

  class_<DenseStringElementsAttr>(m, "DenseStringElementsAttr", denseElsAttr)
      .def(init(&getDenseStringEls<DenseStringElementsAttr>));

  class_<DenseIntOrFPElementsAttr> denseFpOrIntElsAttr{
      m, "DenseIntOrFPElementsAttr", denseElsAttr};

  class_<DenseFPElementsAttr> denseFpElsAttr{
      m, "DenseFPElementsAttr", denseFpOrIntElsAttr};
  denseElsCtorFrom<double>(denseFpElsAttr);
  defBasicIter(denseFpElsAttr);

  class_<DenseIntElementsAttr> denseIntElsAttr{
      m, "DenseIntElementsAttr", denseFpOrIntElsAttr};
  denseElsCtorFrom<int64_t>(denseIntElsAttr);
  denseElsCtorFrom<uint64_t>(denseIntElsAttr);
  defBasicIter(denseIntElsAttr);

  class_<SparseElementsAttr>(m, "SparseElementsAttr", elsAttr)
      .def(init([](ShapedType ty, DenseElementsAttr indices,
                   DenseElementsAttr values) {
        checkType(ty);
        if (!indices) throw std::invalid_argument{"Indices cannot be null"};
        if (!values) throw std::invalid_argument{"Values cannot be null"};
        return SparseElementsAttr::get(ty, indices, values);
      }))
      .def_property_readonly("indices", nullcheck(&SparseElementsAttr::getIndices))
      .def_property_readonly("values", nullcheck(
          overload<DenseElementsAttr(SparseElementsAttr::*)() const>(
              &SparseElementsAttr::getValues)))
      .def("__iter__", nullcheck([](SparseElementsAttr attr) {
        auto range = attr.getValues<Attribute>();
        return make_iterator(range.begin(), range.end());
      }), keep_alive<0, 1>());
      // TODO getIntValues, ... etc.

  implicitly_convertible_from_all<DenseIntOrFPElementsAttr,
      DenseFPElementsAttr, DenseIntElementsAttr>(denseFpOrIntElsAttr);

  implicitly_convertible_from_all<DenseElementsAttr,
      DenseStringElementsAttr, DenseIntOrFPElementsAttr, DenseFPElementsAttr,
      DenseIntElementsAttr>(denseElsAttr);

  implicitly_convertible_from_all<ElementsAttr,
      DenseElementsAttr, DenseStringElementsAttr, DenseIntOrFPElementsAttr,
      DenseFPElementsAttr, DenseIntElementsAttr, SparseElementsAttr>(elsAttr);
}

} // end namespace py
} // end namespace mlir

namespace pybind11 {

template <> struct polymorphic_type_hook<DenseIntOrFPElementsAttr>
    : public polymorphic_type_hooks<DenseIntOrFPElementsAttr,
      DenseFPElementsAttr, DenseIntElementsAttr> {};

template <> struct polymorphic_type_hook<DenseElementsAttr>
    : public polymorphic_type_hooks<DenseElementsAttr,
      DenseStringElementsAttr, DenseIntOrFPElementsAttr, DenseFPElementsAttr,
      DenseIntElementsAttr> {};

template <> struct polymorphic_type_hook<ElementsAttr>
    : public polymorphic_type_hooks<ElementsAttr,
      DenseElementsAttr, DenseStringElementsAttr, DenseIntOrFPElementsAttr,
      DenseFPElementsAttr, DenseIntElementsAttr, SparseElementsAttr> {};

} // end namespace pybind11
