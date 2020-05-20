#include <llvm/Support/raw_ostream.h>
#include <pybind11/detail/common.h>

/// Create a printer for MLIR objects to std::string.
template <typename T>
struct StringPrinter {
  std::string operator()(T t) const {
    std::string buf;
    llvm::raw_string_ostream os{buf};
    t.print(os);
    return std::move(os.str());
  }
};

/// Cast to an overloaded function type.
template <typename FcnT>
auto overload(FcnT fcn) { return fcn; }

/// Move a value to the heap and let Python manage its lifetime.
template <typename T>
std::unique_ptr<T> moveToHeap(T &&t) {
  auto ptr = std::make_unique<T>();
  *ptr = std::move(t);
  return ptr;
}

/// Automatically wrap function calls in a nullcheck of the primary argument.
template <typename FcnT>
std::function<pybind11::detail::function_signature_t<FcnT>>
nullcheck(FcnT fcn, std::string name,
          std::enable_if_t<!std::is_member_function_pointer_v<FcnT>> * = 0) {
  return [fcn, name](auto t, auto ...ts) {
    if (!t)
      throw std::invalid_argument{(name + " is null").c_str()};
    return fcn(t, ts...);
  };
}

/// Automatically wrap member function calls in a nullcheck of the object.
template <typename RetT, typename ObjT, typename... ArgTs>
std::function<RetT(ObjT, ArgTs...)>
nullcheck(RetT(ObjT::*fcn)(ArgTs...), std::string name) {
  return [fcn, name](auto t, ArgTs ...args) {
    if (!t)
      throw std::invalid_argument{(name + " is null").c_str()};
    return (t.*fcn)(args...);
  };
}
