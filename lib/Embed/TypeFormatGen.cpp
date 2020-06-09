#include "dmc/Spec/ParameterList.h"

#include <llvm/Support/SourceMgr.h>

using namespace mlir;
using namespace llvm;
using mlir::dmc::NamedParameter;
using mlir::dmc::NamedParameterRange;

namespace {
class Parameters {
public:
  Parameters(NamedParameterRange parameters)
      : parameters{parameters} {}

private:
  NamedParameterRange parameters;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Format Elements
//===----------------------------------------------------------------------===//

namespace {

class Element {
public:
  virtual ~Element() = default;
};

class VariableElement : public Element {
public:
  explicit VariableElement(NamedParameter var)
      : var{var} {}

private:
  NamedParameter var;
};

class DimensionListDirective : public Element {
public:
  explicit DimensionListDirective(std::unique_ptr<Element> &&element)
      : element{std::move(element)} {}

private:
  std::unique_ptr<Element> element;
};

class LiteralElement : public Element {
public:
  explicit LiteralElement(StringRef literal)
      : literal{literal} {}

private:
  StringRef literal;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Format Lexer
//===----------------------------------------------------------------------===//

namespace {

class Token {
public:
  enum Kind {
    // Markers
    eof,
    error,

    // Stateless
    l_paren,
    r_paren,

    // Keywords
    kw_dims,

    // String valued tokens
    literal,
    variable,
  };

  explicit Token(Kind kind, StringRef spelling)
      : kind{kind},
        spelling{spelling} {}

private:
  Kind kind;
  StringRef spelling;
};

class Lexer {
public:
  explicit Lexer(SourceMgr &mgr, const Parameters &parameters)
      : mgr{mgr},
        parameters{parameters},
        curBuffer{mgr.getMemoryBuffer(mgr.getMainFileID())->getBuffer()},
        curPtr{curBuffer.begin()} {}

  /// Lex one token and return it.
  Token lexToken();

private:
  /// Create a token of the given kind, from the start location to the current
  /// position.
  Token formToken(Token::Kind kind, const char *tokStart) {
    return Token{kind, StringRef(tokStart, curPtr - tokStart)};
  }

  /// Read the next character, skipping lines.
  int getNextChar();

  SourceMgr &mgr;
  const Parameters &parameters;

  StringRef curBuffer;
  const char *curPtr;
};

int Lexer::getNextChar() {
}

} // end anonymous namespace
