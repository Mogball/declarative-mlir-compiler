#include "dmc/Spec/SpecRegion.h"

#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Operation.h>

using namespace mlir;

namespace dmc {

namespace SpecRegion {

bool is(Attribute base) {
  return Any <= base.getKind() && base.getKind() <= NUM_KINDS;
}

} // end namespace SpecRegion

namespace detail {

/// Store the number of blocks expected in the region.
struct SizedRegionAttrStorage : public AttributeStorage {
  using KeyTy = unsigned;

  explicit SizedRegionAttrStorage(KeyTy key) : numBlocks{key} {}
  bool operator==(KeyTy key) const { return key == numBlocks; }
  static llvm::hash_code hashKey(KeyTy key) { return llvm::hash_value(key); }

  static SizedRegionAttrStorage *construct(AttributeStorageAllocator &alloc,
                                           KeyTy key) {
    return new (alloc.allocate<SizedRegionAttrStorage>())
      SizedRegionAttrStorage{key};
  }

  KeyTy numBlocks;
};

/// Store the region constraint applied to all regions captured by a variadic
/// region.
struct VariadicRegionAttrStorage : public AttributeStorage {
  using KeyTy = Attribute;

  explicit VariadicRegionAttrStorage(KeyTy key) : constraint{key} {}
  bool operator==(KeyTy key) const { return key == constraint; }
  static llvm::hash_code hashKey(KeyTy key) { return hash_value(key); }

  static VariadicRegionAttrStorage *construct(AttributeStorageAllocator &alloc,
                                              KeyTy key) {
    return new (alloc.allocate<VariadicRegionAttrStorage>())
      VariadicRegionAttrStorage{key};
  }

  Attribute constraint;
};

} // end namespace detail

/// SizedRegion.
SizedRegion SizedRegion::get(MLIRContext *ctx, unsigned size) {
  return Base::get(ctx, SpecRegion::Sized, size);
}

SizedRegion SizedRegion::getChecked(Location loc, unsigned size) {
  return Base::getChecked(loc, SpecRegion::Sized, size);
}

LogicalResult SizedRegion::verifyConstructionInvariants(Location loc,
                                                        unsigned size) {
  /// numBlocks > 0
  if (size == 0)
    return emitError(loc) << "a region cannot have zero blocks";
  return success();
}

LogicalResult SizedRegion::verify(Region *region) {
  return success(std::size(region->getBlocks()) == getImpl()->numBlocks);
}

/// IsolatedFromAboveRegion.
LogicalResult IsolatedFromAboveRegion::verify(Region *region) {
  return success(region->isIsolatedFromAbove());
}

/// VariadicRegion.
VariadicRegion VariadicRegion::get(Attribute regionConstraint) {
  return Base::get(regionConstraint.getContext(), SpecRegion::Variadic,
                   regionConstraint);
}

VariadicRegion VariadicRegion::getChecked(Location loc,
                                          Attribute regionConstraint) {
  return Base::getChecked(loc, SpecRegion::Variadic, regionConstraint);
}

LogicalResult VariadicRegion::verifyConstructionInvariants(
    Location loc, Attribute regionConstraint) {
  if (!SpecRegion::is(regionConstraint))
    return emitError(loc) << "expected a valid region constraint";
  return success();
}

LogicalResult VariadicRegion::verify(Region *region) {
  return SpecRegion::delegateVerify(getImpl()->constraint, region);
}

/// Region verification.
namespace impl {
LogicalResult verifyRegionConstraints(Operation *op,
                                      mlir::ArrayAttr opRegions) {
  auto regions = op->getRegions();
  auto regionIt = std::begin(regions), regionEnd = std::end(regions);
  auto attrIt = std::begin(opRegions), attrEnd = std::end(opRegions);
  for (unsigned idx = 0; regionIt != regionEnd || attrIt != attrEnd;
       ++attrIt, ++regionIt) {
    /// There can only be one variadic region, and it will always be the last
    /// region. Region counts are verified by another trait.
    assert(attrIt != attrEnd);
    assert(!attrIt->isa<VariadicRegion>() || attrIt == std::prev(attrEnd));
    assert(regionIt != regionEnd || attrIt->isa<VariadicRegion>());
    /// If the current constraint is not variadic, check one region, otherwise,
    /// iterate over the remaining regions and check the variadic constraint.
    for (auto it = regionIt; it != regionEnd &&
         (it == regionIt || attrIt->isa<VariadicRegion>()); ++it, ++idx) {
      if (failed(SpecRegion::delegateVerify(*attrIt, &*it)))
        return op->emitOpError("region #") << idx << " expected " << *attrIt;
    }
  }
  return success();
}
} // end namespace impl

} // end namespace dmc
