#include "Scope.h"
#include "dmc/Embed/InMemoryDef.h"
#include <pybind11/embed.h>

namespace dmc {
namespace py {

InMemoryDef::InMemoryDef(std::string fcnName, std::string fcnSig)
    : os{buf}, pgs{os} {
  pgs.def(fcnName + fcnSig);
}

InMemoryDef::~InMemoryDef() {
  pgs.enddef();
  pybind11::exec(os.str(), getMainScope());
}

} // end namespace py
} // end namespace dmc
