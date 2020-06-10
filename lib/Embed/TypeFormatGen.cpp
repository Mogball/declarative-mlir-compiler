#include "FormatUtils.h"
#include "dmc/Embed/PythonGen.h"
#include "dmc/Spec/ParameterList.h"
#include "dmc/Spec/SpecOps.h"
#include "dmc/Dynamic/DynamicType.h"
#include "dmc/Dynamic/DynamicAttribute.h"

#include <llvm/Support/SourceMgr.h>
#include <llvm/ADT/SmallBitVector.h>

using namespace mlir;
using namespace llvm;
using namespace fmt;
using mlir::dmc::NamedParameter;
using mlir::dmc::NamedParameterRange;
using mlir::dmc::FormatOp;
using ::dmc::py::PythonGenStream;

namespace {
class Parameters : public std::vector<NamedParameter> {
public:
  Parameters(NamedParameterRange parameters)
      : vector{std::begin(parameters), std::end(parameters)} {}

  auto find_by_name(StringRef name) {
    return find_if(*this, [&](auto e) { return e.getName() == name; });
  }
};

namespace FormatElement {
enum Kind {
  ParameterVariable = fmt::Kind::Last,

  DimsDirective,
};
} // end namespace FormatElement

using ParameterVariable =
    VariableElement<NamedParameter, FormatElement::ParameterVariable>;

class DimsDirective : public DirectiveElement<FormatElement::DimsDirective> {
public:
  explicit DimsDirective(std::unique_ptr<Element> var)
      : var{std::move(var)} {}

  auto *getVar() const { return cast<ParameterVariable>(var.get())->getVar(); }

private:
  std::unique_ptr<Element> var;
};

class FormatParser : public fmt::Parser {
public:
  explicit FormatParser(Lexer &lexer, Parameters &parameters)
      : Parser{lexer},
        parameters{parameters},
        seenParameters(parameters.size()) {}

  /// Element parsing.
  LogicalResult parseElement(std::unique_ptr<Element> &element,
                             bool isTopLevel) override;
  /// Directive parsing.
  LogicalResult parseDirective(std::unique_ptr<Element> &element,
                               bool isTopLevel);
  LogicalResult parseDimsDirective(std::unique_ptr<Element> &element,
                                   SMLoc loc);
  /// Variable parsing.
  LogicalResult parseVariable(std::unique_ptr<Element> &element);

  /// Top-level parse and verification.
  LogicalResult parse(std::vector<std::unique_ptr<Element>> &elements);
  /// Verifiers.
  LogicalResult verifyParameters(SMLoc loc);

private:
  Parameters &parameters;
  SmallBitVector seenParameters;
};

LogicalResult FormatParser::parseElement(std::unique_ptr<Element> &element,
                                         bool isTopLevel) {
  if (curToken.isKeyword())
    return parseDirective(element, isTopLevel);
  switch (curToken.getKind()) {
  case Token::literal:
    return parseLiteral(element);
  case Token::variable:
    return parseVariable(element);
  default:
    return emitError(curToken.getLoc(),
                     "expected directive, literal, or variable");
  }
}

LogicalResult FormatParser::parseDirective(std::unique_ptr<Element> &element,
                                           bool isTopLevel) {
  auto dirTok = curToken;
  consumeToken();

  if (!isTopLevel)
    return emitError(dirTok.getLoc(), "directives can only be top-level");
  switch (dirTok.getKind()) {
  case Token::kw_dims:
    return parseDimsDirective(element, dirTok.getLoc());
  default:
    return emitError(dirTok.getLoc(), "unknown directive");
  }
}

LogicalResult FormatParser::parseDimsDirective(
    std::unique_ptr<Element> &element, SMLoc loc) {
  std::unique_ptr<Element> var;
  if (failed(parseToken(Token::l_paren, "expected '(' before argument")) ||
      failed(parseElement(var, false)) ||
      failed(parseToken(Token::r_paren, "expected ')' after argument")))
    return failure();
  if (!isa<ParameterVariable>(var))
    return emitError(loc, "expected a variable as the argument to `dims`");

  element = std::make_unique<DimsDirective>(std::move(var));
  return success();
}

LogicalResult FormatParser::parseVariable(std::unique_ptr<Element> &element) {
  auto varTok = curToken;
  consumeToken();
  auto name = varTok.getSpelling().drop_front();
  if (auto it = parameters.find_by_name(name); it != parameters.end()) {
    auto idx = std::distance(parameters.begin(), it);
    if (seenParameters.test(idx))
      return emitError(varTok.getLoc(), "parameter already bound");
    seenParameters.set(idx);
    element = std::make_unique<ParameterVariable>(&*it);
    return success();
  }
  return emitError(varTok.getLoc(), "unknown parameter name");
}

LogicalResult FormatParser::parse(
    std::vector<std::unique_ptr<Element>> &elements) {
  auto loc = curToken.getLoc();
  auto consumer = [&](std::unique_ptr<Element> element) {
    elements.push_back(std::move(element));
  };
  if (failed(parseElements(std::move(consumer))))
    return failure();

  /// Run verifiers.
  if (failed(verifyParameters(loc)))
    return failure();

  return success();
}

LogicalResult FormatParser::verifyParameters(SMLoc loc) {
  for (auto b = parameters.begin(), e = parameters.end(), it = b; it != e;
       ++it) {
    if (!seenParameters.test(std::distance(b, it)))
      return emitError(loc, "parameter '" + it->getName() + "' not bound");
  }
  return success();
}

class PrinterGen {
public:
  explicit PrinterGen(PythonGenStream &s) : s{s} {}

  void genPrinter(StringRef name,
                  const std::vector<std::unique_ptr<Element>> &elements);
  void genElementPrinter(Element *el);
  void genLiteralPrinter(LiteralElement *el);
  void genVariablePrinter(ParameterVariable *el);
  void genDimsDirectivePrinter(DimsDirective *el);

private:
  PythonGenStream &s;
};

void PrinterGen::genPrinter(
    StringRef name, const std::vector<std::unique_ptr<Element>> &elements) {
  /// Print the type name first so that the parser can distinguish between
  /// different types.
  s.line() << "p.print(\"" << name << "\")";
  for (auto &element : elements) {
    genElementPrinter(element.get());
  }
}

void PrinterGen::genElementPrinter(Element *el) {
  switch (el->getKind()) {
  case Kind::Literal:
    genLiteralPrinter(cast<LiteralElement>(el));
    break;
  case FormatElement::ParameterVariable:
    genVariablePrinter(cast<ParameterVariable>(el));
    break;
  case FormatElement::DimsDirective:
    genDimsDirectivePrinter(cast<DimsDirective>(el));
    break;
  default:
    llvm_unreachable("unknown element kind");
  }
}

void PrinterGen::genLiteralPrinter(LiteralElement *el) {
  s.line() << "p.print(\"" << el->getLiteral() << "\")";
}

void PrinterGen::genVariablePrinter(ParameterVariable *el) {
  s.line() << "p.printAttribute(type.getParameter(\""
      << el->getVar()->getName() << "\"))";
}

void PrinterGen::genDimsDirectivePrinter(DimsDirective *el) {
  s.line() << "p.printDimensionListOrRaw(type.getParameter(\""
      << el->getVar()->getName() << "\"))";
}

} // end anonymous namespace

template <typename OpT, typename DynamicT>
LogicalResult generateTypeFormat(OpT op, DynamicT *impl,
                                 PythonGenStream &parserOs,
                                 PythonGenStream &printerOs) {
  /// Ensure that the string is null-terminated. Ensure it is kept on the stack.
  auto fmtStr = op.getAssemblyFormat().getValue().str();
  SourceMgr mgr;
  mgr.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(fmtStr.c_str()),
                         SMLoc{});

  /// Parse the format.
  Lexer lexer{mgr, op};
  Parameters parameters{impl->getParamSpec()};
  FormatParser parser{lexer, parameters};
  std::vector<std::unique_ptr<Element>> elements;
  if (failed(parser.parse(elements)))
    return failure();

  PrinterGen printerGen{printerOs};
  printerGen.genPrinter(op.getName(), elements);

  parserOs.line() << "pass";

  return success();
}

template LogicalResult generateTypeFormat(
    ::dmc::TypeOp typeOp, ::dmc::DynamicTypeImpl *impl,
    PythonGenStream &parserOs, PythonGenStream &printerOs);
template LogicalResult generateTypeFormat(
    ::dmc::AttributeOp typeOp, ::dmc::DynamicAttributeImpl *impl,
    PythonGenStream &parserOs, PythonGenStream &printerOs);
