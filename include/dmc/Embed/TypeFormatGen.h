#pragma once

#include "PythonGen.h"
#include "dmc/Spec/ParameterList.h"
#include "dmc/Spec/FormatOp.h"

template <typename OpT, typename DynamicT>
mlir::LogicalResult generateTypeFormat(OpT op, DynamicT *impl,
                                       dmc::py::PythonGenStream &parserOs,
                                       dmc::py::PythonGenStream &printerOs);
