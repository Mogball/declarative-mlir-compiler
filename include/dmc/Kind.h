#pragma once

#include <mlir/IR/Types.h>
#include <mlir/IR/Attributes.h>

namespace dmc {
namespace Kind {

constexpr auto FIRST_SPEC_TYPE =
    mlir::Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE;
constexpr auto FIRST_SPEC_ATTR =
    mlir::Attribute::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_ATTR;

constexpr auto FIRST_DYNAMIC_TYPE =
    mlir::Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_1_TYPE;
constexpr auto FIRST_DYNAMIC_ATTR =
    mlir::Attribute::Kind::FIRST_PRIVATE_EXPERIMENTAL_1_ATTR;

} // end namespace Kind
} // end namespace dmc
