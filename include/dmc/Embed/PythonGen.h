#pragma once

#include <llvm/ADT/Twine.h>
#include <llvm/Support/raw_ostream.h>

namespace dmc {
namespace py {

class PythonGenStream {
public:
  class Line {
  public:
    template <typename ArgT> Line &operator<<(ArgT &&arg) & {
      s.os << std::forward<ArgT>(arg);
      return *this;
    }

    template <typename ArgT> Line &&operator<<(ArgT &&arg) && {
      return std::move(operator<<(std::forward<ArgT>(arg)));
    }

    inline Line &operator<<(PythonGenStream &(*fcn)(PythonGenStream &)) & {
      fcn(s);
      return *this;
    }

    inline Line &&operator<<(PythonGenStream &(*fcn)(PythonGenStream &)) && {
      return std::move(operator<<(fcn));
    }

    ~Line();
    Line(Line &&line);

  private:
    explicit Line(PythonGenStream &s);
    Line(const Line &) = delete;

    PythonGenStream &s;
    bool newline;

    friend class PythonGenStream;
  };

  explicit PythonGenStream(llvm::raw_ostream &os);

  Line line();

  PythonGenStream &block(llvm::StringRef ty, llvm::Twine expr);
  PythonGenStream &endblock();

  inline PythonGenStream &if_(llvm::Twine expr) {
    return block("if", expr);
  }
  inline PythonGenStream &else_() {
    endif();
    return block("else", "");
  }
  inline PythonGenStream &def(llvm::Twine decl) {
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
