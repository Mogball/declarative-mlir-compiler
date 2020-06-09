#include "FormatUtils.h"

#include <mlir/IR/Operation.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringSwitch.h>

using namespace llvm;
using namespace mlir;

namespace fmt {

bool isValidLiteral(StringRef value) {
  if (value.empty())
    return false;
  char front = value.front();

  // If there is only one character, this must either be punctuation or a
  // single character bare identifier.
  if (value.size() == 1)
    return isalpha(front) || StringRef("_:,=<>()[]").contains(front);

  // Check the punctuation that are larger than a single character.
  if (value == "->")
    return true;

  // Otherwise, this must be an identifier.
  if (!isalpha(front) && front != '_')
    return false;
  return llvm::all_of(value.drop_front(), [](char c) {
    return isalnum(c) || c == '_' || c == '$' || c == '.';
  });
}

Lexer::Lexer(SourceMgr &mgr, Operation *op)
    : mgr{mgr},
      op{op},
      curBuffer{mgr.getMemoryBuffer(mgr.getMainFileID())->getBuffer()},
      curPtr{curBuffer.begin()} {}

int Lexer::getNextChar() {
  char curChar = *curPtr++;
  switch (curChar) {
  default:
    return static_cast<int>(curChar);
  case 0: {
    // A nul character in the stream is either the end of the current buffer or
    // a random nul in the file. Disambiguate that here.
    if (curPtr - 1 != curBuffer.end())
      return 0;

    // Otherwise, return end of file.
    --curPtr;
    return EOF;
  }
  case '\n':
  case '\r':
    // Handle the newline character by ignoring it and incrementing the line
    // count. However, be careful about 'dos style' files with \n\r in them.
    // Only treat a \n\r or \r\n as a single line.
    if ((*curPtr == '\n' || (*curPtr == '\r')) && *curPtr != curChar)
      ++curPtr;
    return '\n';
  }
}

Token Lexer::lexToken() {
  const char *tokStart = curPtr;

  // This always consumes at least one character.
  int curChar = getNextChar();
  switch (curChar) {
  default:
    // Handle identifiers: [a-zA-Z_]
    if (isalpha(curChar) || curChar == '_')
      return lexIdentifier(tokStart);

    // Unknown character, emit an error.
    return emitError(tokStart, "unexpected character");
  case EOF:
    // Return EOF denoting the end of lexing.
    return formToken(Token::eof, tokStart);

  // Lex punctuation.
  case '^':
    return formToken(Token::caret, tokStart);
  case ',':
    return formToken(Token::comma, tokStart);
  case '=':
    return formToken(Token::equal, tokStart);
  case '?':
    return formToken(Token::question, tokStart);
  case '(':
    return formToken(Token::l_paren, tokStart);
  case ')':
    return formToken(Token::r_paren, tokStart);

  // Ignore whitespace characters.
  case 0:
  case ' ':
  case '\t':
  case '\n':
    return lexToken();

  case '`':
    return lexLiteral(tokStart);
  case '$':
    return lexVariable(tokStart);
  }
}

Token Lexer::lexLiteral(const char *tokStart) {
  assert(curPtr[-1] == '`');

  // Lex a literal surrounded by ``.
  while (const char curChar = *curPtr++) {
    if (curChar == '`')
      return formToken(Token::literal, tokStart);
  }
  return emitError(curPtr - 1, "unexpected end of file in literal");
}

Token Lexer::lexVariable(const char *tokStart) {
  if (!isalpha(curPtr[0]) && curPtr[0] != '_')
    return emitError(curPtr - 1, "expected variable name");

  // Otherwise, consume the rest of the characters.
  while (isalnum(*curPtr) || *curPtr == '_')
    ++curPtr;
  return formToken(Token::variable, tokStart);
}

Token Lexer::lexIdentifier(const char *tokStart) {
  // Match the rest of the identifier regex: [0-9a-zA-Z_\-]*
  while (isalnum(*curPtr) || *curPtr == '_' || *curPtr == '-')
    ++curPtr;

  // Check to see if this identifier is a keyword.
  StringRef str(tokStart, curPtr - tokStart);
  Token::Kind kind =
      llvm::StringSwitch<Token::Kind>(str)
          .Case("attr-dict", Token::kw_attr_dict)
          .Case("attr-dict-with-keyword", Token::kw_attr_dict_w_keyword)
          .Case("functional-type", Token::kw_functional_type)
          .Case("operands", Token::kw_operands)
          .Case("results", Token::kw_results)
          .Case("successors", Token::kw_successors)
          .Case("type", Token::kw_type)
          .Case("dims", Token::kw_dims)
          .Default(Token::identifier);
  return Token(kind, str);
}

Token Lexer::emitError(llvm::SMLoc loc, const Twine &msg) {
  mgr.PrintMessage(loc, llvm::SourceMgr::DK_Error, msg);
  op->emitOpError("in custom assembly format for this operation");
  return formToken(Token::error, loc.getPointer());
}

Token Lexer::emitErrorAndNote(llvm::SMLoc loc, const Twine &msg,
                              const Twine &note) {
  emitError(loc, msg);
  mgr.PrintMessage(loc, llvm::SourceMgr::DK_Note, note);
  return formToken(Token::error, loc.getPointer());
}

Token Lexer::emitError(const char *loc, const Twine &msg) {
  return emitError(llvm::SMLoc::getFromPointer(loc), msg);
}

LogicalResult Parser::parseElements(ElementConsumer &&consumer) {
  // Parse each of the format elements into the main format.
  while (curToken.getKind() != Token::eof) {
    std::unique_ptr<Element> element;
    if (failed(parseElement(element, /*isTopLevel=*/true)))
      return failure();
    consumer(std::move(element));
  }
  return success();
}

LogicalResult Parser::parseLiteral(std::unique_ptr<Element> &element) {
  Token literalTok = curToken;
  consumeToken();

  // Check that the parsed literal is valid.
  StringRef value = literalTok.getSpelling().drop_front().drop_back();
  if (!fmt::isValidLiteral(value))
    return emitError(literalTok.getLoc(), "expected valid literal");

  element = std::make_unique<LiteralElement>(value);
  return success();
}

} // end namespace fmt
