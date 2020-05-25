#include "Expose.h"
#include "Identifier.h"
#include "Context.h"
#include "Support.h"

#include <mlir/IR/Value.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Dialect.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <pybind11/operators.h>

using namespace pybind11;
using namespace mlir;
using namespace llvm;

namespace mlir {
namespace py {

Operation *operationCtor(
    Location loc, const std::string &name, const std::vector<Type> &retTys,
    const std::vector<Value> &operands,
    const std::unordered_map<std::string, Attribute> &attrs,
    const std::vector<Block *> &successors, unsigned numRegions) {
  OperationName opName{name, getMLIRContext()};
  NamedAttrList attrList;
  for (auto &[name, attr] : attrs)
    attrList.push_back({getIdentifierChecked(name), attr});
  return Operation::create(
    loc, opName, retTys, operands, ArrayRef<NamedAttribute>{attrList},
    successors, numRegions);
}

/// Operation * does not need nullcheck because, unlike Value, Type, and
/// Attribute, it does not wrap a nullable underlying type.
void exposeOps(module &m) {
  // All operation instances are managed by MLIR
  class_<Operation, std::unique_ptr<Operation, nodelete>> opCls{m, "Operation"};
  opCls
      .def(init(&operationCtor))
      .def("__repr__", [](Operation *op) {
        std::string buf;
        llvm::raw_string_ostream os{buf};
        op->print(os);
        return std::move(os.str());
      })
      .def_property_readonly("name", [](Operation *op) {
        return op->getName().getStringRef();
      })
      .def("isRegistered", &Operation::isRegistered)
      .def("erase", &Operation::erase)
      .def_property_readonly("block", &Operation::getBlock)
      .def_property_readonly("dialect", &Operation::getDialect)
      .def_property_readonly("loc", &Operation::getLoc)
      .def_property_readonly("parentRegion", &Operation::getParentRegion)
      .def_property_readonly("parentOp", &Operation::getParentOp)
      .def("isProperAncestor", &Operation::isProperAncestor)
      .def("isAncestor", &Operation::isAncestor)
      .def("replaceUsesOfWith", &Operation::replaceUsesOfWith)
      .def("destroy", &Operation::destroy)
      .def("dropAllReferences", &Operation::dropAllReferences)
      .def("dropAllDefinedValueUses", &Operation::dropAllDefinedValueUses)
      .def("moveBefore",
           overload<void(Operation::*)(Operation *)>(&Operation::moveBefore))
      .def("moveAfter",
           overload<void(Operation::*)(Operation *)>(&Operation::moveAfter))
      .def("isBeforeInBlock", &Operation::isBeforeInBlock)
      .def("setOperands", [](Operation *op,
                             const std::vector<Value> &operands) {
        op->setOperands(operands);
      })
      .def("getNumOperands", &Operation::getNumOperands)
      .def("getOperand", &Operation::getOperand)
      .def("setOperand", &Operation::setOperand)
      .def("eraseOperand", &Operation::eraseOperand)
      .def("getOperands", [](Operation *op) {
        return make_iterator(op->operand_begin(), op->operand_end());
      }, keep_alive<0, 1>())
      .def("getOpOperands", [](Operation *op) {
        auto opOperands = op->getOpOperands();
        return make_iterator(std::begin(opOperands), std::end(opOperands));
      }, keep_alive<0, 1>())
      .def("getOpOperand", &Operation::getOpOperand)
      .def("getOperandTypes", [](Operation *op) {
        return make_iterator(op->operand_type_begin(), op->operand_type_end());
      }, keep_alive<0, 1>())
      .def("getNumResults", &Operation::getNumResults)
      .def("getResult", &Operation::getResult)
      .def("getResults", [](Operation *op) {
        return make_iterator(op->result_begin(), op->result_end());
      }, keep_alive<0, 1>())
      .def("getOpResults", [](Operation *op) {
        auto opResults = op->getOpResults();
        return make_iterator(std::begin(opResults), std::end(opResults));
      }, keep_alive<0, 1>())
      .def("getOpResult", &Operation::getOpResult)
      .def("getResultTypes", [](Operation *op) {
        return make_iterator(op->result_type_begin(), op->result_type_end());
      }, keep_alive<0, 1>());
}

} // end namespace py
} // end namespace mlir
