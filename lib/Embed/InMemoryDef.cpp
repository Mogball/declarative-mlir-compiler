#include "Scope.h"
#include "dmc/Embed/InMemoryDef.h"
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

InMemoryClass::InMemoryClass(StringRef clsName, StringRef parentCls,
                             module &m) : m{m} {
  pgs.block("class", clsName + "(" + parentCls + ")");
}

InMemoryClass::~InMemoryClass() {
  pgs.endblock();
  exec(os.str(), m.attr("__dict__"));
}

} // end namespace py
} // end namespace dmc
