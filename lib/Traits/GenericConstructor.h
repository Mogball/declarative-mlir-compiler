#pragma once

#include "dmc/Traits/Registry.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>

namespace dmc {

/// Helpers for unpacking generic argument arrays.
namespace detail {

/// Unpack a typed argument at an index.
template <typename ArgT, unsigned I>
auto unpackArg(llvm::ArrayRef<mlir::Attribute> args) {
  return args[I].cast<ArgT>();
}

/// Call the function with the given arguments.
template <typename... ArgTs, typename FcnT, unsigned... Is>
auto callFcn(FcnT fcn, llvm::ArrayRef<mlir::Attribute> args,
              std::integer_sequence<unsigned, Is...>) {
  return fcn(unpackArg<ArgTs, Is>(args)...);
}

/// Check that an argument at an index is the correct type.
template <typename ArgT, unsigned I>
mlir::LogicalResult checkArgType(mlir::Location loc,
                                 llvm::ArrayRef<mlir::Attribute> args) {
  if (!args[I].isa<ArgT>())
    return mlir::emitError(loc) << "trait constructor expected "
        << typeid(ArgT).name() << " for argument #" << I << " but got "
        << args[I];
  return mlir::success();
}

/// AND a variadic list.
mlir::LogicalResult andFold() { return mlir::success(); }

template <typename... BoolT>
mlir::LogicalResult andFold(mlir::LogicalResult first, BoolT... vals) {
  return mlir::success(mlir::succeeded(first) &&
                       mlir::succeeded(andFold(vals...)));
}

/// Check that the provided argument array has the correct signature.
template <typename... ArgTs, unsigned... Is>
auto checkFcnSignature(mlir::Location loc, llvm::ArrayRef<mlir::Attribute> args,
                       std::integer_sequence<unsigned, Is...>) {
  return andFold(checkArgType<ArgTs, Is>(loc, args)...);
}

} // end namespace detail

template <typename... ArgTs>
class GenericConstructor {
public:
  using ConstructorT = Trait (*)(ArgTs...);
  using Indices = std::make_integer_sequence<unsigned, sizeof...(ArgTs)>;

  GenericConstructor(ConstructorT ctor) : ctor(ctor) {}

  mlir::LogicalResult verifySignature(
      mlir::Location loc, llvm::ArrayRef<mlir::Attribute> args) const {
    if (llvm::size(args) != sizeof...(ArgTs))
      return mlir::emitError(loc) << "expected " << sizeof...(ArgTs)
          << " arguments to trait constructor but got " << llvm::size(args);
    return detail::checkFcnSignature<ArgTs...>(loc, args, Indices{});
  }

  Trait callConstructor(llvm::ArrayRef<mlir::Attribute> args) const {
    return detail::callFcn<ArgTs...>(ctor, args, Indices{});
  }

private:
  ConstructorT ctor;
};

} // end namespace dmc
