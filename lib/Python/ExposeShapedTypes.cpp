#include "Context.h"
#include "Location.h"
#include "Support.h"
#include "Type.h"
#include "Expose.h"

#include <mlir/IR/StandardTypes.h>

using namespace mlir;
using namespace pybind11;

namespace mlir {
namespace py {

template <typename FcnT> auto nullcheck(FcnT fcn) {
  return ::nullcheck(fcn, "type");
}

void ensureRanked(ShapedType ty) {
  if (!ty.hasRank()) throw std::invalid_argument{"ShapedType is not ranked"};
}

void ensureRankedIndex(ShapedType ty, unsigned idx) {
  ensureRanked(ty);
  if (idx >= std::size(ty.getShape()))
    throw index_error{};
}

void exposeShapedTypes(pybind11::module &m, TypeClass &type) {
  /// ShapedType, VectorType, TensorType, RankedTensorTyp, UnrankedTensorType,
  /// BaseMemRefType, MemRefType, UnrankedMemRefType
  class_<ShapedType> shapedTy{m, "ShapedType", type};
  shapedTy
      .def_property_readonly("elementType", nullcheck(&ShapedType::getElementType))
      .def_property_readonly("elementWidth", nullcheck([](ShapedType ty) {
        if (!ty.getElementType().isIntOrIndexOrFloat())
          throw std::invalid_argument{"Element type is not numeric"};
        return ty.getElementTypeBitWidth();
      }))
      .def("__len__", nullcheck([](ShapedType ty) {
        if (!ty.hasStaticShape())
          throw std::invalid_argument{"Shaped type does not have static shape"};
        return ty.getNumElements();
      }))
      .def_property_readonly("rank", nullcheck([](ShapedType ty) {
        ensureRanked(ty);
        return ty.getRank();
      }))
      .def("hasRank", nullcheck(&ShapedType::getRank))
      .def_property_readonly("shape", nullcheck([](ShapedType ty) {
        ensureRanked(ty);
        auto shape = ty.getShape();
        return new std::vector<int64_t>{std::begin(shape), std::end(shape)};
      }))
      .def("hasStaticShape", nullcheck(overload<bool(ShapedType::*)() const>(
            &ShapedType::hasStaticShape)))
      .def_property_readonly("numDynamicDims", nullcheck([](ShapedType ty) {
        ensureRanked(ty);
        return ty.getNumDynamicDims();
      }))
      .def("getDimSize", nullcheck([](ShapedType ty, unsigned idx) {
        ensureRankedIndex(ty, idx);
        return ty.getDimSize(idx);
      }))
      .def("isDynamicDim", nullcheck([](ShapedType ty, unsigned idx) {
        ensureRankedIndex(ty, idx);
        return ty.isDynamicDim(idx);
      }))
      .def("getDynamicDimIndex", nullcheck([](ShapedType ty, unsigned idx) {
        ensureRankedIndex(ty, idx);
        if (!ty.isDynamicDim(idx))
          throw std::invalid_argument{"Dimension at index " + std::to_string(idx) +
                                      " is not dynamic"};
        return ty.getDynamicDimIndex(idx);
      }))
      .def_static("isDynamic", &ShapedType::isDynamic)
      .def_static("isDynamicStrideOrOffset", &ShapedType::isDynamicStrideOrOffset);

  class_<VectorType>(m, "VectorType", shapedTy)
      .def(init([](const std::vector<int64_t> &shape, Type elTy) {
        return VectorType::getChecked(shape, elTy, getUnknownLoc());
      }))
      .def(init([](const std::vector<int64_t> &shape, Type elTy, Location loc) {
        return VectorType::getChecked(shape, elTy, loc);
      }))
      .def_static("isValidElementType", &VectorType::isValidElementType);
}

} // end namespace py
} // end namespace mlir
