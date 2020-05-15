#include "dmc/Spec/ParameterList.h"
#include "dmc/Spec/SpecAttrs.h"
#include "dmc/Spec/Parsing.h"

using namespace dmc;

namespace mlir {
namespace dmc {

#include "dmc/Spec/ParameterList.cpp.inc"

namespace impl {
/// Check that all parameters are SpecAttr.
LogicalResult verifyParameterList(Operation *op, ArrayRef<Attribute> params) {
  unsigned idx = 0;
  for (auto &param : params) {
    if (!SpecAttrs::is(param))
      return op->emitOpError("parameter #") << idx << " expected a SpecAttr "
          << "but got: " << param;
    ++idx;
  }
  return success();
}

void printParameterList(OpAsmPrinter &printer, ArrayRef<Attribute> params) {
  if (!params.empty()) {
    auto it = std::begin(params);
    printer << '<' << *it++;
    for (auto e = std::end(params); it != e; ++it)
      printer << ',' << *it;
    printer << '>';
  }
}
} // end namespace impl

ParseResult ParameterList::parse(OpAsmParser &parser, NamedAttrList &attrList) {
  SmallVector<Attribute, 2> params;
  if (!parser.parseOptionalLess()) {
    do {
      Attribute param;
      if (::dmc::impl::parseSingleAttribute(parser, param))
        return failure();
      /// If a type attribute was provided, wrap in a SpecAttr.
      if (auto tyAttr = param.dyn_cast<mlir::TypeAttr>())
        param = OfTypeAttr::get(tyAttr.getValue());
      params.push_back(param);
    } while (!parser.parseOptionalComma());
    if (parser.parseGreater())
      return failure();
  }
  attrList.append(Trait<int>::getParametersAttrName(),
                  parser.getBuilder().getArrayAttr(params));
  return success();
}

} // end namespace dmc
} // end namespace mlir
