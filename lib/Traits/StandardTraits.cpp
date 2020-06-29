#include "dmc/Traits/StandardTraits.h"

using namespace mlir;

namespace dmc {

LogicalResult HasParent::verifyOp(Operation *op) const {
  auto parentOpName = op->getParentOp()->getName().getStringRef();
  if (parentOpName == parentName)
    return success();
  return op->emitOpError() << "expects parent op '" << parentName
      << "' but got '" << parentOpName << '\'';
}

LogicalResult SingleBlockImplicitTerminator::verifyOp(Operation *op) const {
  /// Each region should have zero or one block, and the block must be
  /// terminated by `terminatorName` op.
  auto regions = op->getRegions();
  unsigned idx{};
  for (auto it = std::begin(regions), e = std::end(regions); it != e;
       ++it, ++idx) {
    auto &region = *it;
    if (region.empty())
      continue;

    if (std::next(std::begin(region)) != std::end(region))
      return op->emitOpError("expects region #") << idx
          << " to have 0 or 1 blocks";

    auto &block = region.front();
    if (block.empty())
      return op->emitOpError("expects a non-empty block");

    auto *term = block.getTerminator();
    if (term->getName().getStringRef() == terminatorName)
      continue;

    return op->emitOpError("expects regions to end with '") << terminatorName
        << "' but found '" << term->getName() << "'";
  }
  return success();
}

static auto unpackTargets(Attribute targets) {
  std::vector<StringRef> ret;
  if (auto name = targets.dyn_cast<StringAttr>()) {
    ret.push_back(name.getValue());
  } else {
    for (auto target : targets.cast<ArrayAttr>())
      ret.push_back(target.cast<StringAttr>().getValue());
  }
  return ret;
}

Alloc::Alloc(Attribute targets)
    : ValueMemoryEffect{unpackTargets(targets)} {}
Free::Free(Attribute targets)
    : ValueMemoryEffect{unpackTargets(targets)} {}
ReadFrom::ReadFrom(Attribute targets)
    : ValueMemoryEffect{unpackTargets(targets)} {}
WriteTo::WriteTo(Attribute targets)
    : ValueMemoryEffect{unpackTargets(targets)} {}

} // end namespace dmc
