#pragma once

#include "dmc/Spec/SpecOps.h"

#include <llvm/Support/raw_ostream.h>

mlir::LogicalResult generateOpFormat(dmc::OperationOp op,
                                     llvm::raw_ostream &parserOs,
                                     llvm::raw_ostream &printerOs);
