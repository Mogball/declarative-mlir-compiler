#include "Context.h"
#include "Identifier.h"
#include "Utility.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Verifier.h>

using namespace pybind11;

namespace mlir {
namespace py {

class ConcreteBuilder : public PatternRewriter {
public:
  explicit ConcreteBuilder() : PatternRewriter{getMLIRContext()} {}
};

static Builder getBuilder() { return Builder{getMLIRContext()}; }

template <typename ValT, typename FcnT>
void def_attr(module &m, const char *name, FcnT fcn) {
  m.def(name, [fcn{std::move(fcn)}](ValT v) {
    return (getBuilder().*fcn)(v);
  });
}

auto builderCreateOp(ConcreteBuilder &builder, object type, pybind11::args args,
                     pybind11::kwargs kwargs) {
  auto ret = type(*args, **kwargs);
  builder.insert(ret.cast<Operation *>());
  return ret;
}

bool operationIsa(Operation *op, object cls) {
  return op->getName().getStringRef() ==
      cls.attr("getName")().cast<std::string>();
}

struct PyPattern {
  PyPattern(object cls, object fcn, unsigned benefit)
      : cls{cls}, fcn{fcn}, benefit{benefit} {}

  object cls, fcn;
  unsigned benefit;
};

struct PyPatternImpl : public RewritePattern {
  explicit PyPatternImpl(PyPattern &pattern)
      : RewritePattern{pattern.cls.attr("getName")().cast<std::string>(),
                       pattern.benefit, getMLIRContext()},
        cls{pattern.cls}, fcn{pattern.fcn} {}

  LogicalResult
  matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto concreteOp = cls(op);
    return success(fcn(concreteOp,
                       static_cast<ConcreteBuilder &>(rewriter)).cast<bool>());
  }

  object cls, fcn;
};

void applyOptPatterns(Operation *op, std::vector<PyPattern> patterns) {
  OwningRewritePatternList patternList;
  for (auto &pattern : patterns) {
    patternList.insert<PyPatternImpl>(pattern);
  }
  applyPatternsAndFoldGreedily(op, patternList);
}

void exposeBuilder(module &m) {
  m.def("I1Type", []() { return getBuilder().getI1Type(); });
  m.def("I8Type", []() { return getBuilder().getIntegerType(8); });
  m.def("I16Type", []() { return getBuilder().getIntegerType(16); });
  m.def("I32Type", []() { return getBuilder().getIntegerType(32); });
  m.def("I64Type", []() { return getBuilder().getIntegerType(64); });

  m.def("I1Attr", [](bool b) {
    return IntegerAttr::get(getBuilder().getI1Type(), b);
  });
  def_attr<int8_t>(m, "I8Attr", &Builder::getI8IntegerAttr);
  def_attr<int16_t>(m, "I16Attr", &Builder::getI16IntegerAttr);
  def_attr<int32_t>(m, "I32Attr", &Builder::getI32IntegerAttr);
  def_attr<int64_t>(m, "I64Attr", &Builder::getI64IntegerAttr);
  def_attr<int64_t>(m, "IndexAttr", &Builder::getIndexAttr);
  def_attr<int32_t>(m, "SI32Attr", &Builder::getSI32IntegerAttr);
  def_attr<uint32_t>(m, "UI32Attr", &Builder::getUI32IntegerAttr);
  def_attr<float>(m, "F16Attr", &Builder::getF16FloatAttr);
  def_attr<float>(m, "F32Attr", &Builder::getF32FloatAttr);
  def_attr<double>(m, "F64Attr", &Builder::getF64FloatAttr);

  def_attr<std::vector<int32_t>>(m, "I32VectorAttr", &Builder::getI32VectorAttr);
  def_attr<std::vector<int64_t>>(m, "I64VectorAttr", &Builder::getI64VectorAttr);
  def_attr<std::vector<int32_t>>(m, "I32ArrayAttr", &Builder::getI32ArrayAttr);
  def_attr<std::vector<int64_t>>(m, "I64ArrayAttr", &Builder::getI64ArrayAttr);
  def_attr<std::vector<int64_t>>(m, "IndexArrayAttr", &Builder::getIndexArrayAttr);
  def_attr<std::vector<float>>(m, "F32ArrayAttr", &Builder::getF32ArrayAttr);
  def_attr<std::vector<double>>(m, "F64ArrayAttr", &Builder::getF64ArrayAttr);
  m.def("StringArrayAttr", [](std::vector<std::string> v) {
    std::vector<StringRef> refs;
    refs.reserve(std::size(v));
    for (auto &s : v)
      refs.emplace_back(s);
    return getBuilder().getStrArrayAttr(refs);
  });

  class_<ConcreteBuilder>(m, "Builder")
      .def(init<>())
      .def("insertBefore",
           overload<void(ConcreteBuilder::*)(Operation *)>(&ConcreteBuilder::setInsertionPoint))
      .def("insertAfter", &ConcreteBuilder::setInsertionPointAfter)
      .def("insertAtStart", &ConcreteBuilder::setInsertionPointToStart)
      .def("insertAtEnd", &ConcreteBuilder::setInsertionPointToEnd)
      .def("getCurrentBlock", &ConcreteBuilder::getInsertionBlock,
           return_value_policy::reference)
      .def("insert", &ConcreteBuilder::insert, return_value_policy::reference)
      .def("create", &builderCreateOp, return_value_policy::reference)
      .def("replace", [](ConcreteBuilder &builder, Operation *op,
                         ValueListRef newValues) {
        builder.replaceOp(op, newValues);
      })
      .def("erase", &ConcreteBuilder::eraseOp)
      .def("erase", &ConcreteBuilder::eraseBlock);

  m.def("verify", [](Operation *op) {
    return succeeded(mlir::verify(op));
  });
  m.def("walkOperations", [](Operation *op, object func) {
    op->walk([func{std::move(func)}](Operation *op) {
      auto ret = func(op);
      if (ret.is(pybind11::cast<pybind11::none>(Py_None)) ||
          ret.cast<bool>()) {
        return WalkResult::advance();
      } else {
        return WalkResult::interrupt();
      }
    });
  });
  m.def("isa", &operationIsa);
  m.def("applyOptPatterns", &applyOptPatterns);

  class_<PyPattern>(m, "Pattern")
      .def(init<object, object, unsigned>(), "cls"_a, "matchFcn"_a,
           "benefit"_a = 0);
}

} // end namespace py
} // end namespace mlir