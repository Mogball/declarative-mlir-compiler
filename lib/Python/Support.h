#include <llvm/Support/raw_ostream.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APFloat.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

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
      throw std::invalid_argument{name + " is null"};
    return fcn(t, ts...);
  };
}

/// Automatically wrap member function calls in a nullcheck of the object.
template <typename RetT, typename ObjT, typename... ArgTs>
std::function<RetT(ObjT, ArgTs...)>
nullcheck(RetT(ObjT::*fcn)(ArgTs...), std::string name) {
  return [fcn, name](auto t, ArgTs ...args) -> RetT {
    if (!t)
      throw std::invalid_argument{name + " is null"};
    return (t.*fcn)(args...);
  };
}

/// For const member functions.
template <typename RetT, typename ObjT, typename... ArgTs>
std::function<RetT(const ObjT, ArgTs...)>
nullcheck(RetT(ObjT::*fcn)(ArgTs...) const, std::string name) {
  return [fcn, name](auto t, ArgTs ...args) -> RetT {
    if (!t)
      throw std::invalid_argument{name + " is null"};
    return (t.*fcn)(args...);
  };
}

/// Create an isa<> check.
template <typename From, typename To>
auto isa() {
  return [](From f) { return f.template isa<To>(); };
}

/// Automatically generate implicit conversions to parent class with
/// LLVM polymorphism: implicit conversion statements and constuctors.
namespace detail {

template <typename, typename...>
struct implicitly_convertible_from_all_helper;

template <typename BaseT, typename FirstT>
struct implicitly_convertible_from_all_helper<BaseT, FirstT> {
  static void doit(pybind11::class_<BaseT> &cls) {
    cls.def(pybind11::init<FirstT>());
    pybind11::implicitly_convertible<FirstT, BaseT>();
  }
};

template <typename BaseT, typename FirstT, typename... DerivedTs>
struct implicitly_convertible_from_all_helper<BaseT, FirstT, DerivedTs...> {
  static void doit(pybind11::class_<BaseT> &cls) {
    implicitly_convertible_from_all_helper<BaseT, FirstT>::doit(cls);
    implicitly_convertible_from_all_helper<BaseT, DerivedTs...>::doit(cls);
  }
};

} // end namespace detail

template <typename BaseT, typename... DerivedTs>
void implicitly_convertible_from_all(pybind11::class_<BaseT> &cls) {
  detail::implicitly_convertible_from_all_helper<
      BaseT, DerivedTs...>::doit(cls);
}

/// Shorthand for declaring polymorphic type hooks for MLIR-style RTTI.
namespace detail {

template <typename, typename...>
struct polymorphic_type_hooks_impl;

template <typename BaseT, typename DerivedT>
struct polymorphic_type_hooks_impl<BaseT, DerivedT> {
  static const void *get(const BaseT *src,
                         const std::type_info *&type) {
    if (src->template isa<DerivedT>()) {
      type = &typeid(DerivedT);
      return static_cast<const DerivedT *>(src);
    }
    return nullptr;
  }
};

template <typename BaseT, typename DerivedT, typename... DerivedTs>
struct polymorphic_type_hooks_impl<BaseT, DerivedT, DerivedTs...> {
  static const void *get(const BaseT *src,
                         const std::type_info *&type) {
    auto ptr = polymorphic_type_hooks_impl<BaseT, DerivedT>::get(src, type);
    return ptr ? ptr :
        polymorphic_type_hooks_impl<BaseT, DerivedTs...>::get(src, type);
  }
};

} // end namespace detail

template <typename BaseT, typename... DerivedTs>
struct polymorphic_type_hooks {
  static const void *get(const BaseT *src,
                         const std::type_info *&type) {
    if (!src || !*src)
      return src;
    return detail::polymorphic_type_hooks_impl<BaseT, DerivedTs...>
          ::get(src, type);
  }
};

/// Common type rebinds.
namespace pybind11 {
namespace detail {

// warning: precision loss ahead
// std::complex<llvm::APInt>
template <> struct type_caster<std::complex<llvm::APInt>> {
  bool load(handle src, bool convert) {
    if (!src)
        return false;
    if (!convert && !PyComplex_Check(src.ptr()))
        return false;
    Py_complex result = PyComplex_AsCComplex(src.ptr());
    if (result.real == -1.0 && PyErr_Occurred()) {
        PyErr_Clear();
        return false;
    }
    /// Store to 64-bit integer
    using storage_t = uint64_t;
    auto realVal = static_cast<storage_t>(result.real);
    auto imagVal = static_cast<storage_t>(result.imag);
    constexpr unsigned bitWidth = sizeof(storage_t) * CHAR_BIT;
    value = std::complex<llvm::APInt>{{bitWidth, llvm::makeArrayRef(realVal)},
                                      {bitWidth, llvm::makeArrayRef(imagVal)}};
    return true;
  }

  static handle cast(const std::complex<llvm::APInt> &src,
                     return_value_policy /* policy */,
                     handle /* parent */) {
    return PyComplex_FromDoubles(src.real().getZExtValue(),
                                 src.imag().getZExtValue());
  }

  PYBIND11_TYPE_CASTER(std::complex<llvm::APInt>, _("complex"));
};

// std::complex<llvm::APFloat>
template <> struct type_caster<std::complex<llvm::APFloat>> {
  bool load(handle src, bool convert) {
    if (!src)
        return false;
    if (!convert && !PyComplex_Check(src.ptr()))
        return false;
    Py_complex result = PyComplex_AsCComplex(src.ptr());
    if (result.real == -1.0 && PyErr_Occurred()) {
        PyErr_Clear();
        return false;
    }
    /// Store to double.
    value = std::complex<llvm::APFloat>{llvm::APFloat{result.real},
                                        llvm::APFloat{result.imag}};
    return true;
  }

  static handle cast(const std::complex<llvm::APFloat> &src,
                     return_value_policy /* policy */,
                     handle /* parent */) {
    return PyComplex_FromDoubles(src.real().convertToDouble(),
                                 src.imag().convertToDouble());
  }

  PYBIND11_TYPE_CASTER(std::complex<llvm::APFloat>, _("complex"));
};

} // end namespace detail
} // end namespace pybind11
