#include "dmc/Python/OpAsm.h"
#include "dmc/Dynamic/DynamicOperation.h"
#include "dmc/Traits/SpecTraits.h"

using namespace mlir;

namespace dmc {
namespace py {

OperationWrap::OperationWrap(Operation *op, DynamicOperation *spec)
    : op{op},
      spec{spec},
      type{spec->getTrait<TypeConstraintTrait>()},
      attr{spec->getTrait<AttrConstraintTrait>()},
      succ{spec->getTrait<SuccessorConstraintTrait>()},
      region{spec->getTrait<RegionConstraintTrait>()} {}

} // end namespace py
} // end namespace dmc
