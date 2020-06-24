#include "Context.h"
#include "Identifier.h"
#include "Utility.h"

#include <mlir/IR/Builders.h>

using namespace pybind11;

namespace mlir {
namespace py {

static Builder getBuilder() { return Builder{getMLIRContext()}; }

template <typename ValT, typename FcnT>
void def_attr(module &m, const char *name, FcnT fcn) {
  m.def(name, [fcn{std::move(fcn)}](ValT v) {
    return (getBuilder().*fcn)(v);
  });
}

auto builderCreateOp(OpBuilder &builder, object type, pybind11::args args,
                     pybind11::kwargs kwargs) {
  auto ret = type(*args, **kwargs);
  builder.insert(ret.cast<Operation *>());
  return ret;
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

  class_<OpBuilder>(m, "Builder")
      .def(init([]() { return OpBuilder{getMLIRContext()}; }))
      .def_static("atStart",
                  [](Block *block) { return OpBuilder::atBlockBegin(block); })
      .def_static("atEnd",
                  [](Block *block) { return OpBuilder::atBlockEnd(block); })
      .def_static("atTerminator",
                  [](Block *block) { return OpBuilder::atBlockTerminator(block); })
      .def("insertBefore",
           overload<void(OpBuilder::*)(Operation *)>(&OpBuilder::setInsertionPoint))
      .def("insertAfter", &OpBuilder::setInsertionPointAfter)
      .def("insertAtStart", &OpBuilder::setInsertionPointToStart)
      .def("insertAtEnd", &OpBuilder::setInsertionPointToEnd)
      .def("getCurrentBlock", &OpBuilder::getInsertionBlock,
           return_value_policy::reference)
      .def("insert", &OpBuilder::insert, return_value_policy::reference)
      .def("create", &builderCreateOp, return_value_policy::reference);

}

} // end namespace py
} // end namespace mlir
