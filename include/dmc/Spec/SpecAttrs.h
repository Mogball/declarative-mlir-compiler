#pragma once

#include "SpecAttrDetail.h"
#include "SpecTypes.h"

#include <mlir/IR/StandardTypes.h>

namespace dmc {

namespace detail {
struct TypedAttrStorage;
} // end namespace detail

namespace SpecAttrs {
enum Kinds {
  Any = mlir::Attribute::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_ATTR,
  Bool,
  Index,
  APInt,

  AnyI,
  I,
  SI,
  UI,

  NUM_ATTRS
};

bool is(mlir::Attribute base);
mlir::LogicalResult delegateVerify(mlir::Attribute base, 
                                   mlir::Attribute attr);

} // end namespace SpecAttrs

class AnyAttr : public SimpleAttr<AnyAttr, SpecAttrs::Any> {
public:
  using Base::Base;
  inline mlir::LogicalResult verify(Attribute) { return mlir::success(); }
};

class BoolAttr : public SimpleAttr<BoolAttr, SpecAttrs::Bool> {
public:
  using Base::Base;
  inline mlir::LogicalResult verify(Attribute attr) {
    return mlir::success(attr.isa<mlir::BoolAttr>());
  }
};

class IndexAttr : public SimpleAttr<IndexAttr, SpecAttrs::Index> {
public:
  using Base::Base;
  inline mlir::LogicalResult verify(Attribute attr) {
    return mlir::success(attr.isa<mlir::IntegerAttr>() &&
        attr.cast<mlir::IntegerAttr>().getType().isa<mlir::IndexType>());
  }
};

class APIntAttr : public SimpleAttr<APIntAttr, SpecAttrs::APInt> {
public:
  using Base::Base;
  inline mlir::LogicalResult verify(Attribute attr) {
    return mlir::success(attr.isa<mlir::IntegerAttr>());
  }
};

class AnyIAttr : public TypedAttrBase<AnyIAttr,
                 SpecTypes::AnyI, mlir::IntegerAttr, AnyIType> {
public:
  using Base::Base;
};

class IAttr : public TypedAttrBase<IAttr,
              SpecTypes::I, mlir::IntegerAttr, IType> {
public:
  using Base::Base;
};

class SIAttr : public TypedAttrBase<SIAttr,
               SpecTypes::SI, mlir::IntegerAttr, SIType> {
public:
  using Base::Base;
};

class UIAttr : public TypedAttrBase<UIAttr,
               SpecTypes::UI, mlir::IntegerAttr, UIType> {
public:
  using Base::Base;
};

/// TODO Default-valued and optional attributes.

} // end namespace dmc
