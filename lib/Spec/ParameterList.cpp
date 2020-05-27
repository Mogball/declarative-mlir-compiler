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
} // end namespace impl

ParseResult ParameterList::parse(OpAsmParser &parser,
                                 NamedAttrList &attrList) {
  /// TypeOp and AttributeOp parameter lists can use type attributes to specify
  /// an OfTypeAttr constraint:
  ///
  ///   <..., i32, ...> => <..., #dmc.OfType<i32>, ...>
  ///
  mlir::ArrayAttr paramAttr;
  auto ofTypeModifier = [&](Attribute attr) -> Attribute {
    if (auto tyAttr = attr.dyn_cast<mlir::TypeAttr>())
      return OfTypeAttr::getChecked(
          parser.getEncodedSourceLoc(parser.getCurrentLocation()),
          tyAttr.getValue());
    return attr;
  };
  if (failed(::dmc::impl::parseOptionalParameterList(parser, paramAttr,
                                                     ofTypeModifier)))
    return failure();
  attrList.append(Trait<int>::getParametersAttrName(), paramAttr);
  return success();
}

} // end namespace dmc
} // end namespace mlir
