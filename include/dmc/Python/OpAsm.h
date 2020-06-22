#pragma once

#include <memory>
#include <mlir/IR/Operation.h>

namespace dmc {
class DynamicOperation;
class TypeConstraintTrait;
class AttrConstraintTrait;
class RegionConstraintTrait;
class SuccessorConstraintTrait;
namespace py {

class OperationWrap {
public:
  explicit OperationWrap(mlir::Operation *op, DynamicOperation *spec);

  auto *getOp() { return op; }
  auto *getSpec() { return spec; }
  auto *getType() { return type; }
  auto *getAttr() { return attr; }
  auto *getSucc() { return succ; }
  auto *getRegion() { return region; }

private:
  mlir::Operation *op;
  DynamicOperation *spec;
  TypeConstraintTrait *type;
  AttrConstraintTrait *attr;
  SuccessorConstraintTrait *succ;
  RegionConstraintTrait *region;
};

} // end namespace py
} // end namespace dmc
