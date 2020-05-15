#ifndef REPARSE_OP_INTERFACE_TD
#define REPARSE_OP_INTERFACE_TD

include "mlir/IR/OpBase.td"

/// An interface that allows operations to reparse themselves to resolve
/// opaque objects.
///
/// This is the second-best solution after incremental parsing: effectively,
/// interpreting MLIR as it is parsed. This would allow for full on-the-fly
/// verification and resolution of symbol references and otherwise opaque
/// values with full error reporting.
///
/// However, the parser API is not amenable to this solution, so a reparsable
/// interface will have to suffice.
def ReparseOpInterface : OpInterface<"ReparseOpInterface"> {
  let description = [{
    Because dynamic objects are registered after first-parse of a MLIR module,
    dynamic types and attributes, for example, remain opaque and must be
    resolved during second pass.

    Lacking a generic interface to traverse operation types and attributes,
    however, the best we can do is ask the Spec operations to reparse parts
    of themselves that might contain opaque objects.
  }];

  let methods = [
    InterfaceMethod<[{
        Request that the operation reparse itself.
      }],
      "ParseResult", "reparse", (ins)
    >,
  ];
}

#endif // REPARSE_OP_INTERFACE_TD
