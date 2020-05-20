#include <llvm/Support/raw_ostream.h>
#include <pybind11/detail/common.h>

template <typename T>
struct StringPrinter {
  std::string operator()(T t) const {
    std::string buf;
    llvm::raw_string_ostream os{buf};
    t.print(os);
    return std::move(os.str());
  }
};

template <typename FcnT>
auto overload(FcnT fcn) { return fcn; }

template <typename T>
std::unique_ptr<T> moveToHeap(T &&t) {
  auto ptr = std::make_unique<T>();
  *ptr = std::move(t);
  return ptr;
}

template <typename FcnT>
std::function<pybind11::detail::function_signature_t<FcnT>>
nullcheck(FcnT fcn, std::string name) {
  return [fcn, name](auto t) {
    if (!t)
      throw std::invalid_argument{(name + " is null").c_str()};
    return fcn(t);
  };
}
