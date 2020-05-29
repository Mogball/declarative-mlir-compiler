#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Location.h>

/// Shorthand for declaring polymorphic type hooks for MLIR-style RTTI.
namespace detail {

template <typename, typename...>
struct polymorphic_type_hooks_impl;

template <typename BaseT, typename DerivedT>
struct polymorphic_type_hooks_impl<BaseT, DerivedT> {
  static const void *get(const BaseT *src,
                         const std::type_info *&type) {
    if (src->template isa<DerivedT>()) {
      type = &typeid(DerivedT);
      return static_cast<const DerivedT *>(src);
    }
    return nullptr;
  }
};

template <typename BaseT, typename DerivedT, typename... DerivedTs>
struct polymorphic_type_hooks_impl<BaseT, DerivedT, DerivedTs...> {
  static const void *get(const BaseT *src,
                         const std::type_info *&type) {
    auto ptr = polymorphic_type_hooks_impl<BaseT, DerivedT>::get(src, type);
    return ptr ? ptr :
        polymorphic_type_hooks_impl<BaseT, DerivedTs...>::get(src, type);
  }
};

} // end namespace detail

template <typename BaseT, typename... DerivedTs>
struct polymorphic_type_hooks {
  static const void *get(const BaseT *src,
                         const std::type_info *&type) {
    if (!src || !*src)
      return src;
    return detail::polymorphic_type_hooks_impl<BaseT, DerivedTs...>
          ::get(src, type);
  }
};

namespace pybind11 {

using namespace mlir;

template <> struct polymorphic_type_hook<Value>
    : public polymorphic_type_hooks<Value,
      BlockArgument, OpResult> {};

template <> struct polymorphic_type_hook<LocationAttr>
    : public polymorphic_type_hooks<LocationAttr,
      UnknownLoc, CallSiteLoc, FileLineColLoc, FusedLoc, NameLoc> {};

/// In order to safely downcast from Location to an impl type, we need to
/// cast the impl pointer directly. Location holds a LocationAttr and we
/// need the pointer to this object to avoid returning a reference to a
/// local value.
template <> struct polymorphic_type_hook<Location> {
  static const void *get(const Location *src,
                         const std::type_info *&type) {
    if (!src)
      return src;
    return polymorphic_type_hook<LocationAttr>::get((*src).operator->(),
                                                    type);
  }
};

template <> struct polymorphic_type_hook<Attribute>
    : public polymorphic_type_hooks<Attribute,
      AffineMapAttr, ArrayAttr, BoolAttr, DictionaryAttr, FloatAttr,
      IntegerAttr, IntegerSetAttr, OpaqueAttr, StringAttr, SymbolRefAttr,
      FlatSymbolRefAttr, TypeAttr,

      ElementsAttr, DenseElementsAttr, DenseStringElementsAttr,
      DenseIntOrFPElementsAttr, DenseFPElementsAttr,
      DenseIntElementsAttr> {};

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

template <> struct polymorphic_type_hook<TensorType>
    : public polymorphic_type_hooks<TensorType,
      RankedTensorType, UnrankedTensorType> {};

template <> struct polymorphic_type_hook<BaseMemRefType>
    : public polymorphic_type_hooks<BaseMemRefType,
      MemRefType, UnrankedMemRefType> {};

template <> struct polymorphic_type_hook<ShapedType>
    : public polymorphic_type_hooks<ShapedType,
      VectorType,
      TensorType, RankedTensorType, UnrankedTensorType,
      BaseMemRefType, MemRefType, UnrankedMemRefType> {};

template <> struct polymorphic_type_hook<Type>
    : public polymorphic_type_hooks<Type,
      FunctionType, OpaqueType,
      ComplexType, IndexType, IntegerType, FloatType, mlir::NoneType,

      VectorType,
      TensorType, RankedTensorType, UnrankedTensorType,
      BaseMemRefType, MemRefType, UnrankedMemRefType,

      TupleType> {};

} // end namespace pybind11
