#pragma once

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APFloat.h>
#include <pybind11/pybind11.h>

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
