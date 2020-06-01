#include "dmc/Spec/SpecSuccessor.h"
#include "dmc/Spec/SpecSuccessorSwitch.h"
#include "dmc/Spec/Parsing.h"

#include <mlir/IR/Builders.h>

using namespace mlir;

namespace dmc {

namespace SpecSuccessor {

bool is(Attribute base) {
  return Any <= base.getKind() && base.getKind() < LAST_SPEC_SUCCESSOR;
}

LogicalResult delegateVerify(Attribute base, Block *block) {
  VerifyAction<Block *> action{block};
  return SpecSuccessor::kindSwitch(action, base);
}

std::string toString(Attribute opSucc) {
  std::string ret;
  llvm::raw_string_ostream os{ret};
  impl::printOpSuccessor(os, opSucc);
  return std::move(os.str());
}

} // end namespace SpecSuccessor

/// VariadicSuccessor.
VariadicSuccessor VariadicSuccessor::getChecked(Location loc,
                                                Attribute succConstraint) {
  return Base::getChecked(loc, SpecSuccessor::Variadic, succConstraint);
}

LogicalResult VariadicSuccessor::verifyConstructionInvariants(
    Location loc, Attribute succConstraint) {
  if (!SpecSuccessor::is(succConstraint))
    return emitError(loc) << "expected a valid successor constraint";
  return success();
}

LogicalResult VariadicSuccessor::verify(Block *block) {
  return SpecSuccessor::delegateVerify(getImpl()->attr, block);
}

/// Parsing
Attribute AnySuccessor::parse(OpAsmParser &parser) {
  return get(parser.getBuilder().getContext());
}

Attribute VariadicSuccessor::parse(OpAsmParser &parser) {
  Attribute opSucc;
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (parser.parseLess() || impl::parseOpSuccessor(parser, opSucc) ||
      parser.parseGreater())
    return {};
  return getChecked(loc, opSucc);
}

void AnySuccessor::print(llvm::raw_ostream &os) {
  os << getName();
}

void VariadicSuccessor::print(llvm::raw_ostream &os) {
  os << getName() << '<';
  impl::printOpSuccessor(os, getImpl()->attr);
  os << '>';
}

} // end namespace dmc
