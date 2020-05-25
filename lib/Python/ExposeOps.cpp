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

void checkOperandIdx(Operation *op, unsigned idx) {
  if (idx >= op->getNumOperands())
    throw index_error{};
}

void checkResultIdx(Operation *op, unsigned idx) {
  if (idx >= op->getNumResults())
    throw index_error{};
}

void checkSuccessorIdx(Operation *op, unsigned idx) {
  if (idx >= op->getNumSuccessors())
    throw index_error{};
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
      .def_property_readonly("block", &Operation::getBlock,
                             return_value_policy::reference)
      .def_property_readonly("dialect", &Operation::getDialect,
                             return_value_policy::reference)
      .def_property_readonly("loc", &Operation::getLoc)
      .def_property_readonly("parentRegion", &Operation::getParentRegion,
                             return_value_policy::reference)
      .def_property_readonly("parentOp", &Operation::getParentOp,
                             return_value_policy::reference)
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
      .def("getOperand", [](Operation *op, unsigned idx) {
        checkOperandIdx(op, idx);
        return op->getOperand(idx);
      })
      .def("setOperand", [](Operation *op, unsigned idx, Value operand) {
        checkOperandIdx(op, idx);
        op->setOperand(idx, operand);
      })
      .def("eraseOperand", [](Operation *op, unsigned idx) {
        checkOperandIdx(op, idx);
        op->eraseOperand(idx);
      })
      .def("getOperands", [](Operation *op) {
        return make_iterator(op->operand_begin(), op->operand_end());
      }, keep_alive<0, 1>())
      .def("getOpOperands", [](Operation *op) {
        auto opOperands = op->getOpOperands();
        return make_iterator(std::begin(opOperands), std::end(opOperands));
      }, keep_alive<0, 1>())
      .def("getOpOperand", [](Operation *op, unsigned idx) {
        checkOperandIdx(op, idx);
        return &op->getOpOperand(idx);
      }, return_value_policy::reference)
      .def("getOperandTypes", [](Operation *op) {
        return make_iterator(op->operand_type_begin(), op->operand_type_end());
      }, keep_alive<0, 1>())
      .def("getNumResults", &Operation::getNumResults)
      .def("getResult", [](Operation *op, unsigned idx) {
        checkResultIdx(op, idx);
        return op->getResult(idx);
      })
      .def("getResults", [](Operation *op) {
        return make_iterator(op->result_begin(), op->result_end());
      }, keep_alive<0, 1>())
      .def("getOpResults", [](Operation *op) {
        auto opResults = op->getOpResults();
        return make_iterator(std::begin(opResults), std::end(opResults));
      }, keep_alive<0, 1>())
      .def("getOpResult", [](Operation *op, unsigned idx) {
        checkResultIdx(op, idx);
        return op->getOpResult(idx);
      })
      .def("getResultTypes", [](Operation *op) {
        return make_iterator(op->result_type_begin(), op->result_type_end());
      }, keep_alive<0, 1>())
      .def("getAttrs", [](Operation *op) {
        auto opAttrs = op->getAttrs();
        std::unordered_map<std::string, Attribute> attrs{std::size(opAttrs)};
        for (auto &[name, attr] : opAttrs) {
          attrs.emplace(name.str(), attr);
        }
        return attrs;
      })
      .def("getAttr", [](Operation *op, const std::string &name) {
        return op->getAttr(name);
      })
      .def("setAttr", [](Operation *op, const std::string &name,
                         Attribute attr) {
        op->setAttr(name, attr);
      })
      .def("removeAttr", [](Operation *op, const std::string &name) {
        return op->removeAttr(name) ==
            MutableDictionaryAttr::RemoveResult::Removed;
      })
      .def("getNumRegions", &Operation::getNumRegions)
      .def("getRegions", [](Operation *op) {
        auto regions = op->getRegions();
        return make_iterator(std::begin(regions), std::end(regions));
      }, keep_alive<0, 1>())
      .def("getRegion", [](Operation *op, unsigned idx) {
        if (idx >= op->getNumRegions())
          throw index_error{};
        return &op->getRegion(idx);
      }, return_value_policy::reference)
      .def("getBlockOperands", [](Operation *op) {
        auto blockArgs = op->getBlockOperands();
        return make_iterator(std::begin(blockArgs), std::end(blockArgs));
      }, keep_alive<0, 1>())
      .def("getSuccessors", [](Operation *op) {
        return make_iterator(op->successor_begin(), op->successor_end());
      }, keep_alive<0, 1>())
      .def("hasSuccessors", &Operation::hasSuccessors)
      .def("getNumSuccessors", &Operation::getNumSuccessors)
      .def("getSuccessor", [](Operation *op, unsigned idx) {
        checkSuccessorIdx(op, idx);
        return op->getSuccessor(idx);
      }, return_value_policy::reference)
      .def("setSuccessor", [](Operation *op, Block *block, unsigned idx) {
        checkSuccessorIdx(op, idx);
        op->setSuccessor(block, idx);
      })
      .def("isCommutative", &Operation::isCommutative)
      .def("isKnownTerminator", &Operation::isKnownTerminator)
      .def("isKnownNonTerminator", &Operation::isKnownNonTerminator)
      .def("isKnownIsolatedFromAbove", &Operation::isKnownIsolatedFromAbove)
      .def("dropAllUses", &Operation::dropAllUses)
      .def("getUses", [](Operation *op) {
        return make_iterator(op->use_begin(), op->use_end());
      }, keep_alive<0, 1>())
      .def("isUsedOutsideOfBlock", &Operation::isUsedOutsideOfBlock)
      .def("hasOneUse", &Operation::hasOneUse)
      .def("useEmpty", &Operation::use_empty)
      .def("getUsers", [](Operation *op) {
        return make_iterator(op->user_begin(), op->user_end());
      }, keep_alive<0, 1>());
}

} // end namespace py
} // end namespace mlir
