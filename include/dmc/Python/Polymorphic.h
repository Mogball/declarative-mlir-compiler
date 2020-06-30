#pragma once

#include "dmc/Dynamic/DynamicType.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Operation.h>

#include <mlir/Dialect/StandardOps/IR/Ops.h>

/// Shorthand for declaring polymorphic type hooks for MLIR-style RTTI.
namespace detail {

template <typename, typename...>
struct polymorphic_type_hooks_impl;

template <typename BaseT, typename DerivedT>
struct polymorphic_type_hooks_impl<BaseT, DerivedT> {
  static const void *get(const BaseT *src, const std::type_info *&type) {
    if (src->template isa<DerivedT>()) {
      type = &typeid(DerivedT);
      // Unsafe: This hack only works because all "subclasses" of Attribute/Type
      // have exactly one member field of ImplType *, where ImplType may differ
      // between subclasses, e.g. DefaultTypeStorage vs. IntegerTypeStorage.
      //
      // It should be done by src->cast<DerivedT>(), however we need to return a
      // pointer value.
      return reinterpret_cast<const DerivedT *>(src);
    }
    return nullptr;
  }
};

template <typename BaseT, typename DerivedT, typename... DerivedTs>
struct polymorphic_type_hooks_impl<BaseT, DerivedT, DerivedTs...> {
  static const void *get(const BaseT *src, const std::type_info *&type) {
    auto ptr = polymorphic_type_hooks_impl<BaseT, DerivedT>::get(src, type);
    return ptr ? ptr :
        polymorphic_type_hooks_impl<BaseT, DerivedTs...>::get(src, type);
  }
};

} // end namespace detail

template <typename BaseT, typename... DerivedTs> struct polymorphic_type_hooks {
  static const void *get(const BaseT *src, const std::type_info *&type) {
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
  static const void *get(const Location *src, const std::type_info *&type) {
    if (!src)
      return src;
    return polymorphic_type_hook<LocationAttr>::get((*src).operator->(), type);
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

      TupleType, dmc::DynamicType> {};

/// mlir::Operation uses MLIR-style RTTI but, unlike Attribute or Type, is not
/// a CTRP base class but is the impl itself. In order to "cast" Operation * to
/// DerivedT, we need to wrap the impl and return a pointer.
namespace detail {

template <typename...>
struct polymorphic_op_impl;

template <typename DerivedT>
struct polymorphic_op_impl<DerivedT> {
  static const void *get(const Operation *src, const std::type_info *&type) {
    // We need to be able to return a pointer
    static thread_local DerivedT op{nullptr};
    if ((op = llvm::dyn_cast<DerivedT>(const_cast<Operation *>(src)))) {
      type = &typeid(DerivedT);
      return &op;
    }
    return nullptr;
  }
};

template <typename DerivedT, typename... DerivedTs>
struct polymorphic_op_impl<DerivedT, DerivedTs...> {
  static const void *get(const Operation *src, const std::type_info *&type) {
    auto ptr = polymorphic_op_impl<DerivedT>::get(src, type);
    return ptr ? ptr : polymorphic_op_impl<DerivedTs...>::get(src, type);
  }
};

} // end namespace detail

template <typename... DerivedTs> struct polymorphic_op_type {
  static const void *get(const Operation *src, const std::type_info *&type) {
    if (!src)
      return src;
    return detail::polymorphic_op_impl<DerivedTs...>::get(src, type);
  }
};

template <> struct polymorphic_type_hook<Operation>
    : public polymorphic_op_type<
      ModuleOp, FuncOp> {};

} // end namespace pybind11
