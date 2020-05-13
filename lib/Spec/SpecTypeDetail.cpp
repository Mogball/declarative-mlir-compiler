#include "dmc/Spec/SpecTypeDetail.h"

#include <unordered_set>

using namespace mlir;

namespace dmc {
namespace impl {

LogicalResult verifyIntWidth(Location loc, unsigned width) {
  switch (width) {
  case 1:
  case 8:
  case 16:
  case 32:
  case 64:
    return success();
  default:
    return emitError(loc) << "integer width must be one of "
        << "[1, 8, 16, 32, 64]";
  }
}

LogicalResult verifyFloatWidth(Location loc, unsigned width) {
  switch (width) {
  case 16:
  case 32:
  case 64:
    return success();
  default:
    return emitError(loc) << "float width must be one of [16, 32, 64]";
  }
}

LogicalResult verifyFloatType(unsigned width, Type ty) {
  switch (width) {
  case 16:
    return success(ty.isF16());
  case 32:
    return success(ty.isF32());
  case 64:
    return success(ty.isF64());
  default:
    llvm_unreachable("Invalid floating point width");
    return failure();
  }
}

LogicalResult verifyWidthList(
    Location loc, ArrayRef<unsigned> widths,
    LogicalResult (&verifyWidth)(Location, unsigned)) {
  /// Check empty list.
  if (widths.empty())
    return emitError(loc) << "empty width list";
  /// Check for duplicate values.
  std::unordered_set<unsigned> widthSet{std::begin(widths), std::end(widths)};
  if (std::size(widthSet) != std::size(widths))
    return emitError(loc) << "duplicate widths in width list";
  /// Verify individual widths.
  for (auto width : widths)
    if (failed(verifyWidth(loc, width)))
      return failure();
  return success();
}

Optional<unsigned> parseSingleWidth(mlir::DialectAsmParser &parser) {
  // *i-type ::= `*I` `<` width `>`
  unsigned width;
  if (parser.parseLess() || parser.parseInteger(width) ||
      parser.parseGreater())
    return {};
  return width;
}

void printSingleWidth(mlir::DialectAsmPrinter &printer, unsigned width) {
  printer << '<' << width << '>';
}

Optional<WidthList> parseWidthList(DialectAsmParser &parser) {
  // *int-of-widths-type ::= `*IntOfWidths` `<` width(`,` width)* `>`
  if (parser.parseLess())
    return {};
  SmallVector<unsigned, 2> widths;
  do {
    unsigned width;
    if (parser.parseInteger(width))
      return {};
    widths.push_back(width);
  } while (!parser.parseOptionalComma());
  if (parser.parseGreater())
    return {};
  return widths;
}

void printWidthList(DialectAsmPrinter &printer, ArrayRef<unsigned> widths) {
  auto it = std::begin(widths);
  printer << '<' << (*it++);
  for (auto e = std::end(widths); it != e; ++it)
    printer << ',' << (*it);
}

} // end namespace impl
} // end namespace dmc
