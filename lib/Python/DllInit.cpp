#include <pybind11/embed.h>

#include "dmc/Python/PyMLIR.h"

using namespace pybind11;

namespace mlir {
namespace py {

static bool inited{false};

void init(MLIRContext *ctx) {}

} // end namespace py
} // end namespace mlir
