#pragma once

#include <llvm/ADT/StringRef.h>

namespace fmt {
/// Returns true if the given string is a valid literal.
bool isValidLiteral(llvm::StringRef value);
} // end namespace fmt
