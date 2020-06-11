#pragma once

#include <pybind11/pybind11.h>

namespace mlir {
namespace py {

template <typename ParserT, typename... ClassTs, typename NameT,
          typename FcnT>
void exposeLiteralParser(pybind11::class_<ParserT, ClassTs...> &cls, NameT name,
                         FcnT fcn) {
  cls.def(name, fcn);
}

template <typename ParserT, typename... ClassTs, typename NameT, typename FcnT,
          typename... TailTs>
void exposeLiteralParser(pybind11::class_<ParserT, ClassTs...> &cls, NameT name,
                         FcnT fcn, TailTs ...tail) {
  exposeLiteralParser(cls, name, fcn);
  exposeLiteralParser(cls, tail...);
}

template <typename ParserT, typename... ClassTs>
void exposeAllLiteralParsers(pybind11::class_<ParserT, ClassTs...> &cls) {
  exposeLiteralParser(
    cls,
    "parseArrow", &ParserT::parseArrow,
    "parseColon", &ParserT::parseColon,
    "parseComma", &ParserT::parseComma,
    "parseEqual", &ParserT::parseEqual,
    "parseLess", &ParserT::parseLess,
    "parseGreater", &ParserT::parseGreater,
    "parseLParen", &ParserT::parseLParen,
    "parseRParen", &ParserT::parseRParen,
    "parseLSquare", &ParserT::parseLSquare,
    "parseRSquare", &ParserT::parseRSquare,
    "parseOptionalArrow", &ParserT::parseOptionalArrow,
    "parseOptionalColon", &ParserT::parseOptionalColon,
    "parseOptionalComma", &ParserT::parseOptionalComma,
    "parseOptionalLess", &ParserT::parseOptionalLess,
    "parseOptionalGreater", &ParserT::parseOptionalGreater,
    "parseOptionalLParen", &ParserT::parseOptionalLParen,
    "parseOptionalRParen", &ParserT::parseOptionalRParen,
    "parseOptionalLSquare", &ParserT::parseOptionalLSquare,
    "parseOptionalRSquare", &ParserT::parseOptionalRSquare,
    "parseOptionalEllipsis", &ParserT::parseOptionalEllipsis);
}

} // end namespace py
} // end namespace mlir
