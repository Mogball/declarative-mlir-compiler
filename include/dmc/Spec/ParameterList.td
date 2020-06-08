#ifndef PARAMETER_LIST_TD
#define PARAMETER_LIST_TD

include "mlir/IR/OpBase.td"

/// Interface for operations that have a parameter list of attributes.
def ParameterList : OpInterface<"ParameterList"> {
  let description = [{
    This class provides an interface and verifier for operations that take
    a list of SpecAttr parameters: <attr0, attr1, ..., attrN>.
  }];

  let methods = [
    InterfaceMethod<[{
        Get the Op's parameter list.
      }],
      "NamedParameterRange", "getParameters", (ins), [{}], [{
        auto attr = this->getOperation()->template
            getAttrOfType<mlir::ArrayAttr>(getParametersAttrName());
        NamedParameter (*unwrap)(Attribute) = [](Attribute attr) {
          return attr.cast<NamedParameter>();
        };
        return llvm::map_range(attr.getValue(), unwrap);
      }]
    >,
    InterfaceMethod<[{
        Replace the Op's parameter list.
      }],
      "void", "setParameters", (ins "ArrayRef<Attribute>":$params), [{}], [{
        auto *op = this->getOperation();
        op->setAttr(getParametersAttrName(),
                    ArrayAttr::get(params, op->getContext()));
      }]
    >,
    InterfaceMethod<[{
        Print the parameter list.
      }],
      "void", "printParameters", (ins "OpAsmPrinter &":$printer), [{}], [{
        auto params = dyn_cast<ConcreteOp>(this->getOperation()).getParameters();
        if (!params.empty()) {
          printer << '<';
          llvm::interleaveComma(params, printer, [&](Attribute attr) {
            auto param = attr.cast<NamedParameter>();
            printer << param.getName() << ": " << param.getConstraint();
          });
          printer << '>';
        }
      }]
     >,
  ];

  let verify = [{
    auto paramsAttr = $_op->getAttrOfType<ArrayAttr>(getParametersAttrName());
    if (!paramsAttr)
      return $_op->emitOpError("expected an ArrayAttr named: ")
          << getParametersAttrName();
    return success();
  }];

  let extraTraitClassDeclaration = [{
    static llvm::StringLiteral getParametersAttrName() { return "params"; }
  }];

  /// Parser and printer for parameter lists.
  let extraClassDeclaration = [{
    static ParseResult parse(OpAsmParser &parser, NamedAttrList &attrList);
  }];
}

#endif // PARAMETER_LIST_TD
