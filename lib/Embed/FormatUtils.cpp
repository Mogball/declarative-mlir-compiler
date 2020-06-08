#include "FormatUtils.h"

#include <llvm/ADT/StringExtras.h>

using namespace llvm;

namespace fmt {

bool isValidLiteral(StringRef value) {
  if (value.empty())
    return false;
  char front = value.front();

  // If there is only one character, this must either be punctuation or a
  // single character bare identifier.
  if (value.size() == 1)
    return isalpha(front) || StringRef("_:,=<>()[]").contains(front);

  // Check the punctuation that are larger than a single character.
  if (value == "->")
    return true;

  // Otherwise, this must be an identifier.
  if (!isalpha(front) && front != '_')
    return false;
  return llvm::all_of(value.drop_front(), [](char c) {
    return isalnum(c) || c == '_' || c == '$' || c == '.';
  });
}

} // end namespace fmt
