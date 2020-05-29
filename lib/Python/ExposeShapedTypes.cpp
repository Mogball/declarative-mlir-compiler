#include "Context.h"
#include "Location.h"
#include "Utility.h"
#include "Type.h"
#include "Expose.h"

#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/AffineMap.h>

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
      .def(init([](const std::vector<int64_t> &shape, Type elTy, Location loc) {
        return VectorType::getChecked(shape, elTy, loc);
      }), "shape"_a, "elementType"_a, "location"_a = getUnknownLoc())
      .def_static("isValidElementType", &VectorType::isValidElementType);

  class_<TensorType> tensorTy{m, "TensorType", shapedTy};
  tensorTy.def_static("isValidElementType", &TensorType::isValidElementType);

  class_<RankedTensorType>(m, "RankedTensorType", tensorTy)
      .def(init([](const std::vector<int64_t> &shape, Type elTy, Location loc) {
        return RankedTensorType::getChecked(shape, elTy, loc);
      }), "shape"_a, "elementType"_a, "location"_a = getUnknownLoc());

  class_<UnrankedTensorType>(m, "UnrankedTensorType", tensorTy)
      .def(init(&UnrankedTensorType::getChecked),
           "elementType"_a, "location"_a = getUnknownLoc());

  class_<BaseMemRefType> baseMemRefTy{m, "BaseMemRefType", shapedTy};

  class_<MemRefType>(m, "MemRefType", baseMemRefTy)
      .def(init([](const std::vector<int64_t> &shape, Type elTy,
                   const std::vector<AffineMap> &affineMapComposition,
                   unsigned memorySpace, Location loc) {
        return MemRefType::getChecked(shape, elTy, affineMapComposition,
                                      memorySpace, loc);
      }), "shape"_a, "elementType"_a,
          "affineMapComposition"_a = std::vector<AffineMap>{},
          "memorySpace"_a = 0, "location"_a = getUnknownLoc())
      .def_property_readonly("affineMaps", nullcheck([](MemRefType ty) {
        auto affineMaps = ty.getAffineMaps();
        return new std::vector<AffineMap>{std::begin(affineMaps),
                                          std::end(affineMaps)};
      }))
      .def_property_readonly("memorySpace",
                             nullcheck(&MemRefType::getMemorySpace));

  class_<UnrankedMemRefType>(m, "UnrankedMemRefType", baseMemRefTy)
      .def(init(&UnrankedMemRefType::getChecked),
           "elementType"_a, "memorySpace"_a, "location"_a = getUnknownLoc())
      .def_property_readonly("memorySpace",
                             nullcheck(&UnrankedMemRefType::getMemorySpace));

  implicitly_convertible_from_all<TensorType,
      RankedTensorType, UnrankedTensorType>(tensorTy);

  implicitly_convertible_from_all<BaseMemRefType,
      MemRefType, UnrankedMemRefType>(baseMemRefTy);

  implicitly_convertible_from_all<ShapedType,
      VectorType,
      TensorType, RankedTensorType, UnrankedTensorType,
      BaseMemRefType, MemRefType, UnrankedMemRefType>(shapedTy);
}

} // end namespace py
} // end namespace mlir
