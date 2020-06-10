#ifndef FORMAT_OP_TD
#define FORMAT_OP_TD

include "mlir/IR/OpBase.td"

def FormatOp : OpInterface<"FormatOp"> {
  let description = [{
    This class provides an interface and verifier for operations that define
    MLIR objects with an optional format string, for generating parsers and
    printers.
  }];

  let methods = [
    InterfaceMethod<[{ Get the format string attribute. }],
      "StringAttr", "getAssemblyFormat", (ins), [{}], [{
        return this->getOperation()->template getAttrOfType<StringAttr>(
            getFmtAttrName());
      }]
    >
  ];

  let verify = [{
    auto fmtAttr = $_op->getAttr(getFmtAttrName());
    if (fmtAttr && !fmtAttr.isa<StringAttr>())
      return $_op->emitOpError("expected attribute '") << getFmtAttrName()
          << "' to be a string attribute";
    return success();
  }];

  let extraTraitClassDeclaration = [{
    static llvm::StringLiteral getFmtAttrName() { return "fmt"; }
  }];
}

#endif // FORMAT_OP_TD
