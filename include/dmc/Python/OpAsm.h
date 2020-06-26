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

  mlir::Value getOperand(std::string name);
  mlir::Value getResult(std::string name);
  mlir::Value getOperandOrResult(std::string name);

  mlir::ValueRange getOperandGroup(std::string name);
  mlir::ValueRange getResultGroup(std::string name);
  mlir::ValueRange getOperandOrResultGroup(std::string name);

  mlir::Region &getRegion(std::string name);

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
