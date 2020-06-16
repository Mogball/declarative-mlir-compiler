#pragma once

#include <mlir/Support/LogicalResult.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/SMLoc.h>
#include <llvm/Support/SourceMgr.h>

namespace mlir {
class Operation;
}

namespace fmt {

/// Basic element kinds.
namespace Kind {
enum {
  Literal,
  Optional,

  Last
};
} // end namespace Kind

/// Returns true if the given string is a valid literal.
bool isValidLiteral(llvm::StringRef value);

/// Returns a literal parser call.
std::string getParserForLiteral(llvm::StringRef value, bool optional);

/// This class represents a single format element.
class Element {
public:
  explicit Element(unsigned kind) : kind{kind} {}
  virtual ~Element() = default;

  /// Return the kind of this element.
  unsigned getKind() const { return kind; }

private:
  /// The kind of this element.
  unsigned kind;
};

/// This class represents an instance of an variable element. A variable refers
/// to something registered on the operation itself, e.g. an argument, result,
/// etc.
template <typename VarT, unsigned kindVal>
class VariableElement : public Element {
public:
  explicit VariableElement(const VarT *var)
      : Element{kindVal}, var{var} {}

  static bool classof(const Element *element) {
    return element->getKind() == kindVal;
  }

  const VarT *getVar() { return var; }

protected:
  const VarT *var;
};

/// This class implements single kind directives.
template <unsigned type>
class DirectiveElement : public Element {
public:
  explicit DirectiveElement()
      : Element{type} {};

  static bool classof(const Element *ele) {
    return ele->getKind() == type;
  }
};

/// This class represents an instance of a literal element.
class LiteralElement : public Element {
public:
  explicit LiteralElement(llvm::StringRef literal)
      : Element{Kind::Literal},
        literal{literal} {}

  static bool classof(const Element *element) {
    return element->getKind() == Kind::Literal;
  }

  /// Return the literal for this element.
  llvm::StringRef getLiteral() const { return literal; }

private:
  /// The spelling of the literal for this element.
  llvm::StringRef literal;
};

/// This class represents a group of elements that are optionally emitted based
/// upon an optional variable of the operation.
class OptionalElement : public Element {
public:
  explicit OptionalElement(std::vector<std::unique_ptr<Element>> &&elements,
                           unsigned anchor)
      : Element{Kind::Optional},
        elements{std::move(elements)},
        anchor{anchor} {}

  static bool classof(const Element *element) {
    return element->getKind() == Kind::Optional;
  }

  /// Return the nested elements of this grouping.
  auto getElements() const { return llvm::make_pointee_range(elements); }

  /// Return the anchor of this optional group.
  Element *getAnchor() const { return elements[anchor].get(); }

private:
  /// The child elements of this optional.
  std::vector<std::unique_ptr<Element>> elements;
  /// The index of the element that acts as the anchor for the optional group.
  unsigned anchor;
};

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

/// This class represents a specific token in the input format.
class Token {
public:
  enum Kind {
    // Markers.
    eof,
    error,

    // Tokens with no info.
    l_paren,
    r_paren,
    caret,
    comma,
    equal,
    question,

    // Keywords.
    keyword_start,
    kw_attr_dict,
    kw_attr_dict_w_keyword,
    kw_functional_type,
    kw_operands,
    kw_results,
    kw_successors,
    kw_type,
    kw_symbol,
    kw_dims,
    keyword_end,

    // String valued tokens.
    identifier,
    literal,
    variable,
  };

  explicit Token(unsigned kind, llvm::StringRef spelling)
      : kind{kind},
        spelling{spelling} {}

  /// Return the bytes that make up this token.
  llvm::StringRef getSpelling() const { return spelling; }

  /// Return the kind of this token.
  unsigned getKind() const { return kind; }

  /// Return a location for this token.
  llvm::SMLoc getLoc() const {
    return llvm::SMLoc::getFromPointer(spelling.data());
  }

  /// Return if this token is a keyword.
  bool isKeyword() const { return kind > keyword_start && kind < keyword_end; }

private:
  /// Discriminator that indicates the kind of token this is.
  unsigned kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  llvm::StringRef spelling;
};

/// This class implements a simple lexer for operation assembly format strings.
class Lexer {
public:
  explicit Lexer(llvm::SourceMgr &mgr, mlir::Operation *op);

  Token lexToken();

  /// Emit an error to the lexer with the given location and message.
  Token emitError(llvm::SMLoc loc, const llvm::Twine &msg);

  Token emitError(const char *loc, const llvm::Twine &msg);
  Token emitErrorAndNote(llvm::SMLoc loc, const llvm::Twine &msg,
                         const llvm::Twine &note);

private:
  /// Return the next character in the stream.
  int getNextChar();

  /// Lex an identifier, literal, or variable.
  Token lexIdentifier(const char *tokStart);
  Token lexLiteral(const char *tokStart);
  Token lexVariable(const char *tokStart);

protected:
  Token formToken(unsigned kind, const char *tokStart) {
    return Token{kind, llvm::StringRef(tokStart, curPtr - tokStart)};
  }

  llvm::SourceMgr &mgr;
  mlir::Operation *op;

private:
  llvm::StringRef curBuffer;
  const char *curPtr;
};

/// This class implements a parser for an assembly format.
class Parser {
public:
  explicit Parser(Lexer &lexer)
      : lexer{lexer},
        curToken{lexer.lexToken()} {}

  /// Parse all function.
  using ElementConsumer = std::function<void(std::unique_ptr<Element>)>;
  mlir::LogicalResult parseElements(ElementConsumer &&consumer);

  /// Literal parsing.
  mlir::LogicalResult parseLiteral(std::unique_ptr<Element> &element);

protected:
  virtual mlir::LogicalResult parseElement(std::unique_ptr<Element> &element,
                                           bool isTopLevel) = 0;

  /// Advance the current lexer onto the next token.
  void consumeToken() {
    assert(curToken.getKind() != Token::eof &&
           curToken.getKind() != Token::error &&
           "shouldn't advance past EOF or errors");
    curToken = lexer.lexToken();
  }

  /// Parse an expected token.
  mlir::LogicalResult parseToken(Token::Kind kind, const llvm::Twine &msg) {
    if (curToken.getKind() != kind)
      return emitError(curToken.getLoc(), msg);
    consumeToken();
    return mlir::success();
  }

  /// Error handling.
  mlir::LogicalResult emitError(llvm::SMLoc loc, const llvm::Twine &msg) {
    lexer.emitError(loc, msg);
    return mlir::failure();
  }
  mlir::LogicalResult emitErrorAndNote(llvm::SMLoc loc, const llvm::Twine &msg,
                                       const llvm::Twine &note) {
    lexer.emitErrorAndNote(loc, msg, note);
    return mlir::failure();
  }

  /// Fields.
  Lexer &lexer;
  Token curToken;
};

} // end namespace fmt
