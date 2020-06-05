#pragma once

#include <memory>

namespace mlir {
class Operation;
struct OperationState;
} // end namespace mlir

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

class ResultWrap {
public:
  explicit ResultWrap(mlir::OperationState &result)
      : result{result} {}

  auto &getResult() { return result; }

private:
  mlir::OperationState &result;
};

} // end namespace py
} // end namespace dmc
