#include "Scope.h"
#include "dmc/Embed/InMemoryDef.h"

#include <llvm/ADT/StringSwitch.h>
#include <pybind11/embed.h>

using namespace llvm;
using namespace pybind11;

namespace dmc {
namespace py {

InMemoryDef::InMemoryDef(StringRef fcnName, StringRef fcnSig) {
  pgs.def(fcnName + fcnSig);
}

InMemoryDef::~InMemoryDef() {
  pgs.enddef();
  // Store the parser/printer in the internal scope
  exec(os.str(), getInternalScope());
}

InMemoryClass::InMemoryClass(StringRef clsName, ArrayRef<StringRef> parentCls,
                             module &m) : m{m} {
  // intercept invalid class names
  auto valid = StringSwitch<bool>(clsName)
      .Case("return", false)
      .Case("def", false)
      .Case("class", false)
      .Case("assert", false)
      .Default(true);
  auto name = clsName.str();
  if (!valid)
    name.front() = std::toupper(name.front());

  auto line = pgs.line() << "class " << name << "(";
  llvm::interleaveComma(parentCls, line, [&](StringRef cls) { line << cls; });
  line << "):" << incr;
}

InMemoryClass::~InMemoryClass() {
  pgs.endblock();
  exec(os.str(), m.attr("__dict__"));
}

} // end namespace py
} // end namespace dmc
