#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>

namespace dmc {
class DynamicDialect;
namespace py {
void exposeDialectInternal(DynamicDialect *dialect,
                           llvm::ArrayRef<llvm::StringRef> scope);
} // end namespace py
} // end namespace dmc
