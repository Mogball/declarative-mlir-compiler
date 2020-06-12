#include "Utility.h"
#include "AsmUtils.h"

#include <mlir/IR/DialectImplementation.h>
#include <llvm/ADT/STLExtras.h>
#include <pybind11/pybind11.h>

using namespace pybind11;
using namespace mlir;
using namespace llvm;

namespace dmc {
namespace py {
extern void exposeTypeWrap(module &m);
} // end namespace py
} // end namespace dmc

namespace mlir {
namespace py {

namespace {
static void printDimensionListOrRaw(DialectAsmPrinter &p, Attribute attr) {
  if (auto arr = attr.dyn_cast<ArrayAttr>()) {
    interleave(arr, p, [&](Attribute el) {
      if (auto i = el.dyn_cast<IntegerAttr>();
          i && i.getValue().getSExtValue() == -1) {
        p << "?";
      } else {
        p << el;
      }
    }, "x");
    p << "x";
  } else {
    p.printAttribute(attr);
  }
}
} // end anonymous namespace

void exposeDialectAsm(module &m) {
  class_<DialectAsmPrinter, std::unique_ptr<DialectAsmPrinter, nodelete>>
      (m, "DialectAsmPrinter")
      .def("print", [](DialectAsmPrinter &p, std::string val) {
        p << val;
      })
      .def("printAttribute", &DialectAsmPrinter::printAttribute)
      .def("printDimensionListOrRaw", &printDimensionListOrRaw);

  class_<DialectAsmParser, std::unique_ptr<DialectAsmParser, nodelete>>
      parserCls{m, "DialectAsmParser"};
  exposeAllLiteralParsers(parserCls);
  parserCls
      .def("parseDimensionList", [](DialectAsmParser &parser,
                                    bool allowDynamic) {
        SmallVector<int64_t, 4> dims;
        if (failed(parser.parseDimensionList(dims, allowDynamic)))
          return make_tuple(nullptr, false);
        std::vector<int64_t> ret{std::begin(dims), std::end(dims)};
        return pybind11::make_tuple(std::move(ret), true);
      })
      .def("parseAttribute", [](DialectAsmParser &parser) {
        Attribute attr;
        auto ret = parser.parseAttribute(attr);
        return make_tuple(attr, ret);
      });

  dmc::py::exposeTypeWrap(m);
}

} // end namespace py
} // end namespace mlir
