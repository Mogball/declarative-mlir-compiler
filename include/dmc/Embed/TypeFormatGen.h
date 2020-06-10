#pragma once

#include "PythonGen.h"
#include "dmc/Spec/ParameterList.h"
#include "dmc/Spec/FormatOp.h"

mlir::LogicalResult generateTypeFormat(mlir::dmc::NamedParameterRange params,
                                       mlir::dmc::FormatOp op,
                                       dmc::py::PythonGenStream &parserOs,
                                       dmc::py::PythonGenStream &printerOs);
