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
      "ArrayRef<Attribute>", "getParameters", (ins), [{}], [{
        return this->getOperation()->template getAttrOfType<mlir::ArrayAttr>(
            getParametersAttrName()).getValue();
      }]
    >,
  ];

  let verify = [{
    auto paramsAttr = $_op->getAttrOfType<ArrayAttr>(getParametersAttrName());
    if (!paramsAttr)
      return $_op->emitOpError("expected an ArrayAttr named: ")
          << getParametersAttrName();
    return impl::verifyParameterList($_op, paramsAttr.getValue());
  }];

  let extraTraitClassDeclaration = [{
    static llvm::StringLiteral getParametersAttrName() { return "params"; }
  }];
}

#endif // PARAMETER_LIST_TD
