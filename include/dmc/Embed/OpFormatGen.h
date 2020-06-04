#pragma once

#include "PythonGen.h"
#include "dmc/Spec/SpecOps.h"

#include <llvm/Support/raw_ostream.h>

mlir::LogicalResult generateOpFormat(dmc::OperationOp op,
                                     dmc::py::PythonGenStream &parserOs,
                                     dmc::py::PythonGenStream &printerOs);
