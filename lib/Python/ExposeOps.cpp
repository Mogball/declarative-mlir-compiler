#include "Expose.h"
#include "Identifier.h"
#include "Context.h"
#include "Utility.h"

#include <mlir/IR/Value.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Dialect.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <pybind11/operators.h>

using namespace pybind11;
using namespace mlir;
using dmc::BaseOp;

namespace mlir {
namespace py {

Operation *operationCtor(
    Location loc, const std::string &name, TypeListRef retTys,
    ValueListRef operands, AttrDictRef attrs, BlockListRef successors,
    unsigned numRegions) {
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

namespace detail {
template <typename...> struct op_implicit_conversion_helper;
template <typename OpTy> struct op_implicit_conversion_helper<OpTy> {
  template <typename ClassT> static void doit(ClassT &cls) {
    cls.def(init([](OpTy op) -> Operation * { return op; }),
            return_value_policy::reference);
    implicitly_convertible<OpTy, Operation>();
  }
};
template <typename OpTy, typename... OpTys>
struct op_implicit_conversion_helper<OpTy, OpTys...> {
  template <typename ClassT> static void doit(ClassT &cls) {
    op_implicit_conversion_helper<OpTy>::doit(cls);
    op_implicit_conversion_helper<OpTys...>::doit(cls);
  }
};
} // end namespace detail

template <typename... OpTys, typename ClassT>
void op_implicit_conversion(ClassT &cls) {
  detail::op_implicit_conversion_helper<OpTys...>::doit(cls);
}

template <typename RetT, typename... ArgTs>
auto wrapBase(RetT(Operation::*fcn)(ArgTs...)) {
  return [fcn](Operation *op, ArgTs ...args) -> RetT {
    return (op->*fcn)(std::forward<ArgTs>(args)...);
  };
}

template <typename T, typename FcnT,
          std::enable_if_t<std::is_same_v<T, Operation>, int> = 0>
auto wrap(FcnT fcn) {
  return fcn;
}
template <typename T, typename FcnT,
          std::enable_if_t<std::is_same_v<T, BaseOp>, int> = 0>
auto wrap(FcnT fcn) {
  return wrapBase(fcn);
}

template <typename T, typename... ExtraTs>
void defOpMethods(class_<T, ExtraTs...> &cls) {
  cls
      .def("__repr__", [](Operation *op) {
        std::string buf;
        llvm::raw_string_ostream os{buf};
        op->print(os);
        return std::move(os.str());
      })
      .def_property_readonly("name", [](Operation *op) {
        return op->getName().getStringRef().str();
      })
      .def("isRegistered", wrap<T>(&Operation::isRegistered))
      .def("erase", wrap<T>(&Operation::erase))
      .def_property_readonly("block", wrap<T>(&Operation::getBlock),
                             return_value_policy::reference)
      .def_property_readonly("dialect", wrap<T>(&Operation::getDialect),
                             return_value_policy::reference)
      .def_property_readonly("loc", wrap<T>(&Operation::getLoc))
      .def_property_readonly("parentRegion",
                             wrap<T>(&Operation::getParentRegion),
                             return_value_policy::reference)
      .def_property_readonly("parentOp", wrap<T>(&Operation::getParentOp),
                             return_value_policy::reference)
      .def("isProperAncestor", wrap<T>(&Operation::isProperAncestor))
      .def("isAncestor", wrap<T>(&Operation::isAncestor))
      .def("replaceUsesOfWith", wrap<T>(&Operation::replaceUsesOfWith))
      .def("destroy", wrap<T>(&Operation::destroy))
      .def("dropAllReferences", wrap<T>(&Operation::dropAllReferences))
      .def("dropAllDefinedValueUses",
           wrap<T>(&Operation::dropAllDefinedValueUses))
      .def("moveBefore",
           wrap<T>(overload<void(Operation::*)(Operation *)>(
               &Operation::moveBefore)))
      .def("moveAfter",
           wrap<T>(overload<void(Operation::*)(Operation *)>(
               &Operation::moveAfter)))
      .def("isBeforeInBlock", wrap<T>(&Operation::isBeforeInBlock))
      .def("setOperands", [](Operation *op, ValueListRef operands) {
        op->setOperands(operands);
      })
      .def("getNumOperands", wrap<T>(&Operation::getNumOperands))
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
      .def("getNumResults", wrap<T>(&Operation::getNumResults))
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
      .def("getNumRegions", wrap<T>(&Operation::getNumRegions))
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
      .def("hasSuccessors", wrap<T>(&Operation::hasSuccessors))
      .def("getNumSuccessors", wrap<T>(&Operation::getNumSuccessors))
      .def("getSuccessor", [](Operation *op, unsigned idx) {
        checkSuccessorIdx(op, idx);
        return op->getSuccessor(idx);
      }, return_value_policy::reference)
      .def("setSuccessor", [](Operation *op, Block *block, unsigned idx) {
        checkSuccessorIdx(op, idx);
        op->setSuccessor(block, idx);
      })
      .def("isCommutative", wrap<T>(&Operation::isCommutative))
      .def("isKnownTerminator", wrap<T>(&Operation::isKnownTerminator))
      .def("isKnownNonTerminator", wrap<T>(&Operation::isKnownNonTerminator))
      .def("isKnownIsolatedFromAbove",
           wrap<T>(&Operation::isKnownIsolatedFromAbove))
      .def("dropAllUses", wrap<T>(&Operation::dropAllUses))
      .def("getUses", [](Operation *op) {
        return make_iterator(op->use_begin(), op->use_end());
      }, keep_alive<0, 1>())
      .def("isUsedOutsideOfBlock", wrap<T>(&Operation::isUsedOutsideOfBlock))
      .def("hasOneUse", wrap<T>(&Operation::hasOneUse))
      .def("useEmpty", wrap<T>(&Operation::use_empty))
      .def("getUsers", [](Operation *op) {
        return make_iterator(op->user_begin(), op->user_end());
      }, keep_alive<0, 1>());

}

Block *regionAddEntryBlock(Region &region, TypeListRef types) {
  auto *entry = new Block{};
  region.push_back(entry);
  entry->addArguments(types);
  return entry;
}

/// Operation * does not need nullcheck because, unlike Value, Type, and
/// Attribute, it does not wrap a nullable underlying type.
void exposeOps(module &m) {
  // All operation instances are managed by MLIR
  class_<BaseOp> opCls{m, "Op"};
  defOpMethods(opCls);
  opCls.def(init<Operation *>());

  class_<Operation, std::unique_ptr<Operation, nodelete>> baseCls{m,
                                                                  "Operation"};
  defOpMethods(baseCls);
  baseCls
      .def(init([](BaseOp &base) -> Operation * { return base; }),
           return_value_policy::reference)
      .def(init(&operationCtor), return_value_policy::reference);
  implicitly_convertible<BaseOp, Operation>();
  implicitly_convertible<Operation, BaseOp>();

  class_<Region, std::unique_ptr<Region, nodelete>>(m, "Region")
      .def("empty", &Region::empty)
      .def("getBlock", [](Region &region, unsigned idx) {
        auto it = std::next(std::begin(region), idx);
        if (it == std::end(region))
          throw index_error{};
        return &*it;
      }, return_value_policy::reference_internal)
      .def("append", &Region::push_back)
      .def("push_front", &Region::push_front)
      .def("addEntryBlock", &regionAddEntryBlock)
      .def("__len__", [](Region &region) -> unsigned {
        return std::distance(region.begin(), region.end());
      });

  class_<Block, std::unique_ptr<Block, nodelete>>(m, "Block")
      // Block must be given to a region or else this will leak
      .def(init([]() { return new Block; }), return_value_policy::reference)
      .def("getNumArguments", &Block::getNumArguments)
      .def("getArgument", [](Block &block, unsigned i) {
        if (i >= block.getNumArguments())
          throw index_error{};
        return block.getArgument(i);
      })
      .def("getArguments", [](Block &block) {
        return make_iterator(block.args_begin(), block.args_end());
      }, keep_alive<0, 1>())
      .def("addArgs", [](Block &block, TypeListRef types) {
        block.addArguments(types);
      })
      .def("addArg", [](Block &block, Type ty) {
        block.addArguments(ty);
      });

  exposeModule(m, opCls);

  op_implicit_conversion<ModuleOp>(baseCls);
}

} // end namespace py
} // end namespace mlir

namespace pybind11 {
template <> struct polymorphic_type_hook<BaseOp>
    : public polymorphic_type_hooks<BaseOp,
      ModuleOp> {};
} // end namespace pybind11
