#include "dmc/Python/OpAsm.h"
#include "dmc/Dynamic/DynamicOperation.h"
#include "dmc/Traits/SpecTraits.h"

#include <pybind11/pytypes.h>
#include <pybind11/pybind11.h>

using namespace pybind11;
using namespace mlir;

namespace dmc {
namespace py {

namespace {

template <typename SizedT, typename SameT, typename GetSingleFcn>
ValueRange getGroup(OperationWrap &op, unsigned idx, GetSingleFcn getSingle) {
  if (op.getSpec()->getTrait<SizedT>()) {
    return SizedT::getGroup(op.getOp(), idx);
  } else if (op.getSpec()->getTrait<SameT>()) {
    return SameT::getGroup(op.getOp(), idx);
  } else {
    return getSingle(op.getOp(), idx);
  }
}

ValueRange getOperandGroup(OperationWrap &op, unsigned idx) {
  return getGroup<dmc::SizedOperandSegments, dmc::SameVariadicOperandSizes>(
      op, idx, [](Operation *op, unsigned idx) {
        auto it = std::next(op->operand_begin(), idx);
        return OperandRange{it, std::next(it)};
      });
}

ValueRange getResultGroup(OperationWrap &op, unsigned idx) {
  return getGroup<dmc::SizedResultSegments, dmc::SameVariadicResultSizes>(
      op, idx, [](Operation *op, unsigned idx) {
        auto it = std::next(op->result_begin(), idx);
        return ResultRange{it, std::next(it)};
      });
}

Value getOperand(OperationWrap &op, unsigned idx) {
  return *getOperandGroup(op, idx).begin();
}

Value getResult(OperationWrap &op, unsigned idx) {
  return *getResultGroup(op, idx).begin();
}

template <typename GetFcn, typename NamedTypeRange>
std::result_of_t<GetFcn(OperationWrap &, unsigned)>
getValueOrGroup(OperationWrap &op, GetFcn getVal, std::string name,
                NamedTypeRange values) {
  unsigned idx{};
  for (auto value : values) {
    if (value.name == name)
      return getVal(op, idx);
    ++idx;
  }
  throw std::invalid_argument{"Unable to find named value '" + name +
                              "' for op '" + op.getSpec()->getName() + "'"};
}

} // end anonymous namespace

OperationWrap::OperationWrap(Operation *op, DynamicOperation *spec)
    : op{op},
      spec{spec},
      type{spec->getTrait<TypeConstraintTrait>()},
      attr{spec->getTrait<AttrConstraintTrait>()},
      succ{spec->getTrait<SuccessorConstraintTrait>()},
      region{spec->getTrait<RegionConstraintTrait>()} {}

void exposeOperationWrap(module &m) {
  class_<ValueRange>(m, "ValueRange")
      .def("getTypes", [](ValueRange &values) {
        list types;
        for (auto value : values)
          types.append(value.getType());
        return types;
      })
      .def("__len__", [](ValueRange &values) {
        return llvm::size(values);
      })
      .def("__getitem__", [](ValueRange &values, unsigned idx) {
        return *std::next(std::begin(values), idx);
      });

  class_<OperationWrap>(m, "OperationWrap")
      .def(init([](Operation *op) {
        return OperationWrap{op, DynamicOperation::of(op)};
      }))
      .def("getName", [](OperationWrap &op) {
        return op.getOp()->getName().getStringRef().str();
      })
      .def("getAttrs", [](OperationWrap &op) {
        return pybind11::cast(op.getOp()).attr("getAttrs")().cast<dict>();
      })
      .def("getAttr", [](OperationWrap &op, std::string name) {
        return op.getOp()->getAttr(name);
      })
      .def("getOperandTypes", [](OperationWrap &op) {
        list types;
        for (auto type : op.getOp()->getOperandTypes())
          types.append(pybind11::cast(type));
        return types;
      })
      .def("getResultTypes", [](OperationWrap &op) {
        list types;
        for (auto type : op.getOp()->getResultTypes())
          types.append(pybind11::cast(type));
        return types;
      })
      .def("getOperand", [](OperationWrap &op, std::string name) {
        return getValueOrGroup(op, &getOperand, std::move(name),
                               op.getType()->getOpType().getOperands());
      })
      .def("getResult", [](OperationWrap &op, std::string name) {
        return getValueOrGroup(op, &getResult, std::move(name),
                               op.getType()->getOpType().getResults());
      })
      .def("getOperandGroup", [](OperationWrap &op, std::string name) {
        return getValueOrGroup(op, &getOperandGroup, std::move(name),
                               op.getType()->getOpType().getOperands());
      })
      .def("getResultGroup", [](OperationWrap &op, std::string name) {
        return getValueOrGroup(op, &getResultGroup, std::move(name),
                               op.getType()->getOpType().getResults());
      })
      .def("getOperands", [](OperationWrap &op) -> ValueRange {
        return op.getOp()->getOperands();
      })
      .def("getRegion", [](OperationWrap &op, std::string name) {
        unsigned idx{};
        for (auto &region : op.getRegion()->getOpRegions().getRegions()) {
          if (region.name == name) {
            return &op.getOp()->getRegion(idx);
          }
          ++idx;
        }
        throw std::invalid_argument{"Unable to find a region named '" + name +
                                    "' for operation '" +
                                    op.getSpec()->getName() + "'"};
      }, return_value_policy::reference_internal);
}

void exposeResultWrap(module &m) {
  class_<OperationState>(m, "OperationState")
      .def("addTypes", [](OperationState &result, list types) {
        for (auto type : types)
          result.types.push_back(type.cast<Type>());
      })
      .def("addRegion", [](OperationState &result) {
        return result.addRegion();
      }, return_value_policy::reference);
  class_<NamedAttrList>(m, "NamedAttrList");
}

} // end namespace py
} // end namespace dmc
