#pragma once

#include <llvm/Support/raw_ostream.h>

namespace dmc {
namespace py {

class PythonGenStream {
public:
  class Line {
  public:
    template <typename ArgT> Line &operator<<(ArgT &&arg) {
      s.os << std::forward<ArgT>(arg);
      return *this;
    }

    inline Line &operator<<(PythonGenStream &(*fcn)(PythonGenStream &)) {
      fcn(s);
      return *this;
    }

  private:
    explicit Line(PythonGenStream &s);
    ~Line();

    PythonGenStream &s;

    friend class PythonGenStream;
  };

  explicit PythonGenStream(llvm::raw_ostream &os);

  Line line();

  PythonGenStream &block(llvm::StringRef ty, llvm::StringRef expr);
  PythonGenStream &endblock();

  inline PythonGenStream &if_(llvm::StringRef expr) {
    return block("if", expr);
  }
  inline PythonGenStream &else_() {
    endif();
    return block("else", "");
  }
  inline PythonGenStream &def(llvm::StringRef decl) {
    return block("def", decl);
  }

  inline PythonGenStream &endif() { return endblock(); }
  inline PythonGenStream &enddef() { return endblock(); }

  inline PythonGenStream &incr() {
    changeIndent(4);
    return *this;
  }
  inline PythonGenStream &decr() {
    changeIndent(-4);
    return *this;
  }

private:
  void changeIndent(int delta);

  llvm::raw_ostream &os;
  int indent;

  friend class Line;
  friend PythonGenStream &incr(PythonGenStream &);
  friend PythonGenStream &decr(PythonGenStream &);
};

inline PythonGenStream &incr(PythonGenStream &s) { return s.incr(); }
inline PythonGenStream &decr(PythonGenStream &s) { return s.decr(); }

} // end namespace py
} // end namespace dmc
