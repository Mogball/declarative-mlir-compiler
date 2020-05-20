#include <llvm/Support/raw_os_ostream.h>

template <typename T>
std::ostream &printToOs(std::ostream &os, T &&t) {
  llvm::raw_os_ostream rawOs{os};
  std::forward<T>(t).print(rawOs);
  return os;
}

template <typename FcnT>
auto overload(FcnT fcn) { return fcn; }

template <typename T>
T *moveToHeap(T &&t) {
  auto *ptr = new T;
  *ptr = std::move(t);
  return ptr;
}
