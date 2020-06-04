#pragma once

#include "PythonGen.h"

namespace dmc {
namespace py {

class InMemoryDef {
public:
  explicit InMemoryDef(std::string fcnName, std::string fcnSig);
  ~InMemoryDef();

  inline PythonGenStream &stream() { return pgs; }
  inline const std::string &str() { return os.str(); }

private:
  std::string buf;
  llvm::raw_string_ostream os;
  PythonGenStream pgs;
};

} // end namespace py
} // end namespace dmc
