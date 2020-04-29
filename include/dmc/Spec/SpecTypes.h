#pragma once

namespace dmc {

/// SpecDialect types aim to capture type matching and verification logic.
/// E.g. !dmc.Int will verify the concrete type with ty.is<IntegerType>() and
/// !dmc.AnyOf<$Types...> will assert that the concrete type matches one of
/// the specified allowed types.
///
/// Variadic operands or results are specified with !dmc.Variadic<$Type>.
/// The following restrictions apply:
/// - Non-variadic values must preceed all variadic values
/// - If there are any variadic compound types, then the Op must have a
///   variadic size specifier OpTrait:
///
///     !dmc.Variadic<i32>, !dmc.Variadic<f32> is OK as the types
///     are mutually exclusive, but
///
///     !dmc.Variadic<!dmc.Float>, !dmc.Variadic<f32> requires a variadic size
///     specifier (SameVariadicSize) as the types may not be mutually exclusive.
///
namespace detail {
} // end namespace detail

} // end namespace dmc
