#include <mlir/IR/Attributes.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;

template <typename ClassT>
constexpr unsigned get_kind() {
  for (unsigned kind = 0;; ++kind) {
    if (ClassT::kindof(kind))
      return kind;
  }
  return 0;
}

int main() {
  auto kind = get_kind<StringAttr>();
  llvm::errs() << kind << "\n";

  return 0;
}
