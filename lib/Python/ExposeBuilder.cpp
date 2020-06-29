#include "Context.h"
#include "Identifier.h"
#include "Utility.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Transforms/DialectConversion.h>

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
  PyPattern(object cls, object fcn, list generated, unsigned benefit)
      : cls{cls}, fcn{fcn}, generated{generated}, benefit{benefit} {}

  object cls, fcn;
  list generated;
  unsigned benefit;
};

static ArrayRef<StringRef> getGeneratedOps(list generated) {
  // Need to keep references alive until RewritePattern constructor is done
  thread_local std::vector<std::string> names;
  thread_local std::vector<StringRef> ret;
  ret.clear();
  names.clear();
  ret.reserve(generated.size());
  names.reserve(generated.size());
  for (auto cls : generated) {
    names.push_back(cls.attr("getName")().cast<std::string>());
    ret.push_back(names.back());
  }
  return ret;
}

struct PyPatternImpl : public RewritePattern {
  explicit PyPatternImpl(PyPattern &pattern)
      : RewritePattern{pattern.cls.attr("getName")().cast<std::string>(),
                       py::getGeneratedOps(pattern.generated),
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

static auto getPatternList(std::vector<PyPattern> patterns) {
  OwningRewritePatternList patternList;
  for (auto &pattern : patterns) {
    patternList.insert<PyPatternImpl>(pattern);
  }
  return patternList;
}

bool applyOptPatterns(Operation *op, std::vector<PyPattern> patterns) {
  auto patternList = getPatternList(std::move(patterns));
  return applyPatternsAndFoldGreedily(op, patternList);
}

bool applyPartialConversion(Operation *op, std::vector<PyPattern> patterns,
                            ConversionTarget target) {
  auto patternList = getPatternList(std::move(patterns));
  return succeeded(applyPartialConversion(op, target, patternList, nullptr));
}

bool applyFullConversion(Operation *op, std::vector<PyPattern> patterns,
                         ConversionTarget target) {
  auto patternList = getPatternList(std::move(patterns));
  return succeeded(applyFullConversion(op, target, patternList, nullptr));
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
      .def(init<object, object, list, unsigned>(), "cls"_a, "matchFcn"_a,
           "generatedOps"_a = list{}, "benefit"_a = 0);

  class_<ConversionTarget>(m, "ConversionTarget")
      .def(init([]() { return new ConversionTarget{*getMLIRContext()}; }))
      .def("addLegalOp", [](ConversionTarget &target, object cls) {
        auto name = cls.attr("getName")().cast<std::string>();
        OperationName opName{name, getMLIRContext()};
        target.setOpAction(opName,
                           ConversionTarget::LegalizationAction::Legal);
      })
      .def("addIllegalOp", [](ConversionTarget &target, object cls) {
        auto name = cls.attr("getName")().cast<std::string>();
        OperationName opName{name, getMLIRContext()};
        target.setOpAction(opName,
                           ConversionTarget::LegalizationAction::Illegal);
      })
      .def("addLegalDialect", [](ConversionTarget &target, object cls) {
        auto name = cls.attr("name").cast<std::string>();
        target.addLegalDialect(name);
      })
      .def("addIllegalDialect", [](ConversionTarget &target, object cls) {
        auto name = cls.attr("name").cast<std::string>();
        target.addIllegalDialect(name);
      });
}

} // end namespace py
} // end namespace mlir
