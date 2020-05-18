#include "dmc/Spec/Parsing.h"

#include <mlir/IR/Builders.h>

using namespace mlir;

namespace dmc {
namespace impl {

/// Parse a single attribute. OpAsmPrinter provides only an API to parse
/// a NamedAttribute into a list.
ParseResult parseSingleAttribute(OpAsmParser &parser, Attribute &attr) {
  NamedAttrList attrList;
  return parser.parseAttribute(attr, "single", attrList);
}

/// Parse an optional parameter list. If not found, returns success() and an
/// empty array attribute.
///
/// parameter-list ::= (`<` attr (`,` attr)* `>`)?
ParseResult parseOptionalParameterList(OpAsmParser &parser,
                                       mlir::ArrayAttr &attr) {
  return parseOptionalParameterList(parser, attr,
                                    [](Attribute attr) { return attr; });
}

/// Parse an optional parameter list with an on-the-fly modifier. Callers can
/// use the modifier to intercept certain parameters without re-traversing the
/// resultant attribute array.
ParseResult parseOptionalParameterList(
    OpAsmParser &parser, mlir::ArrayAttr &attr,
    Attribute (modifier)(Attribute)) {
  SmallVector<Attribute, 2> params;
  if (!parser.parseOptionalLess()) {
    do {
      Attribute param;
      if (parseSingleAttribute(parser, param))
        return failure();
      params.push_back(modifier(param));
    } while (!parser.parseOptionalComma());
    if (parser.parseGreater())
      return failure();
  }
  attr = parser.getBuilder().getArrayAttr(params);
  return success();
}

} // end namespace impl
} // end namespace dmc
