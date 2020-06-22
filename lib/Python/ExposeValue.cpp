#include "Utility.h"
#include "Expose.h"
#include "Identifier.h"
#include "Context.h"

#include <mlir/IR/Value.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Block.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <pybind11/operators.h>

using namespace pybind11;
using namespace mlir;
using namespace llvm;

namespace mlir {
namespace py {

template <typename FcnT> auto nullcheck(FcnT fcn) {
  return ::nullcheck(fcn, "value");
}

template <typename DerivedT, typename ValueType>
class_<DerivedT> getIROperandClass(
    module &m, const char *name) {
  class_<DerivedT> cls{m, name};
  cls
      .def(init<Operation *>())
      .def(init<Operation *, ValueType>())
      .def("get", &DerivedT::get)
      .def("set", &DerivedT::set)
      .def("is", &DerivedT::is)
      .def_property_readonly(
        "owner", overload<Operation *(DerivedT::*)()>(&DerivedT::getOwner),
        return_value_policy::reference)
      .def("drop", &DerivedT::drop)
      .def("getNextOperandUsingThisValue",
           &DerivedT::getNextOperandUsingThisValue);
  return cls;
}

template <typename OperandType>
class_<IRObjectWithUseList<OperandType>> getIRObjectWithUseListClass(
    module &m, const char *name) {
  using Base = IRObjectWithUseList<OperandType>;
  class_<Base> cls{m, name};
  cls
      .def("dropAllUses", &Base::dropAllUses)
      .def("replaceAllUsesWith", &Base::replaceAllUsesWith)
      .def("getUses", [](Base *list) {
        return make_iterator(list->use_begin(), list->use_end());
      }, keep_alive<0, 1>())
      .def("getUsers", [](Base *list) {
        return make_iterator(list->user_begin(), list->user_end());
      }, keep_alive<0, 1>())
      .def("hasOneUse", &Base::hasOneUse)
      .def("useEmpty", &Base::use_empty);
  return cls;
}

void exposeValue(module &m) {
  class_<Value> value{m, "Value"};
  value
      .def(init([]() { return Value{nullptr}; }))
      .def(init<const Value &>())
      .def("__bool__", &Value::operator bool)
      .def("__repr__", StringPrinter<Value>{})
      .def("__hash__", overload<hash_code(Value)>(&hash_value))
      .def(self == self)
      .def(self != self)
      .def("getType", nullcheck(&Value::getType))
      .def_property("type", nullcheck(&Value::getType),
                    nullcheck(&Value::setType))
      .def_property_readonly("definingOp", nullcheck(
          overload<Operation *(Value::*)() const>(&Value::getDefiningOp)),
          return_value_policy::reference)
      .def_property_readonly("loc", nullcheck(&Value::getLoc))
      .def_property_readonly("parentRegion", nullcheck(&Value::getParentRegion),
          return_value_policy::reference)
      .def_property_readonly("parentBlock", nullcheck(&Value::getParentBlock),
          return_value_policy::reference)
      .def_property_readonly("useList", nullcheck(&Value::getUseList),
          return_value_policy::reference)
      .def("dropAllUses", nullcheck(&Value::dropAllUses))
      .def("replaceAllUsesWith", nullcheck(&Value::replaceAllUsesWith))
      .def("replaceAllUsesExcept", nullcheck([](
            Value newValue,
            const std::unordered_set<Operation *> &exceptions) {
              SmallPtrSet<Operation *, 4> ptrSet{std::begin(exceptions),
                                                 std::end(exceptions)};
              newValue.replaceAllUsesExcept(newValue, ptrSet);
            }))
      // TODO replaceAllUsesWithIf when calling into Python is available
      .def("isUsedOutsideOfBlock", nullcheck(&Value::isUsedOutsideOfBlock))
      .def("getUses", nullcheck([](Value val) {
        return make_iterator(val.use_begin(), val.use_end());
      }), keep_alive<0, 1>())
      .def("useEmpty", nullcheck(&Value::use_empty))
      .def("hasOneUse", nullcheck(&Value::hasOneUse))
      .def("getUsers", nullcheck([](Value val) {
        return make_iterator(val.user_begin(), val.user_end());
      }), keep_alive<0, 1>())
      .def_property_readonly("kind", nullcheck(&Value::getKind));

  class_<BlockArgument>(m, "BlockArgument", value)
      .def_property_readonly("owner", nullcheck(&BlockArgument::getOwner))
      .def_property_readonly("argNumber",
                             nullcheck(&BlockArgument::getArgNumber));

  class_<OpResult>(m, "OpResult", value)
      .def_property_readonly("owner", nullcheck(&OpResult::getOwner))
      .def_property_readonly("resultNumber",
                             nullcheck(&OpResult::getResultNumber));

  auto blockOperand = getIROperandClass<BlockOperand, Block *>(m,
                                                               "BlockOperand");
  blockOperand
      .def_property_readonly("useList", &BlockOperand::getUseList,
                             return_value_policy::reference)
      .def_property_readonly("operandNumber", &BlockOperand::getOperandNumber);

  auto opOperand = getIROperandClass<OpOperand, detail::OpaqueValue>(
      m, "OpOperand");
  opOperand
      .def_property_readonly("useList", &OpOperand::getUseList,
                             return_value_policy::reference)
      .def_property_readonly("operandNumber", &OpOperand::getOperandNumber)
      .def("set", &OpOperand::set)
      .def("get", &OpOperand::get);

  getIRObjectWithUseListClass<BlockOperand>(m, "BlockArgumentUseList");
  getIRObjectWithUseListClass<OpOperand>(m, "OperandUseList");

  implicitly_convertible_from_all<
      BlockArgument, OpResult>(value);
}

} // end namespace py
} // end namespace mlir
