#pragma once

#include "dmc/Spec/ParameterList.h"

#include <mlir/IR/Attributes.h>
#include <llvm/ADT/ArrayRef.h>

namespace dmc {
class DynamicType;
class DynamicAttribute;
namespace py {
class TypeWrap {
public:
  explicit TypeWrap(DynamicType type);
  explicit TypeWrap(DynamicAttribute attr);

  auto getParams() { return params; }
  auto getSpec() { return paramSpec; }

private:
  llvm::ArrayRef<mlir::Attribute> params;
  NamedParameterRange paramSpec;
};
} // end namespace py
} // end namespace dmc
