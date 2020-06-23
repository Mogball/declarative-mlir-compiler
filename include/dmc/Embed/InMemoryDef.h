#pragma once

#include "PythonGen.h"

#include <llvm/ADT/ArrayRef.h>

namespace pybind11 {
class module;
}

namespace dmc {
namespace py {

class InMemoryStream {
public:
  inline PythonGenStream &stream() { return pgs; }
  inline const std::string &str() { return os.str(); }

protected:
  std::string buf;
  llvm::raw_string_ostream os{buf};
  PythonGenStream pgs{os};
};

class InMemoryDef : public InMemoryStream {
public:
  explicit InMemoryDef(llvm::StringRef fcnName, llvm::StringRef fcnSig);
  ~InMemoryDef();
};

class InMemoryClass : public InMemoryStream {
public:
  explicit InMemoryClass(
      llvm::StringRef clsName, llvm::ArrayRef<llvm::StringRef> parentCls,
      pybind11::module &m);
  ~InMemoryClass();

private:
  pybind11::module &m;
};

} // end namespace py
} // end namespace dmc
