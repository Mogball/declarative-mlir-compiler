#include "PythonGen.h"

using namespace llvm;

namespace dmc {
namespace py {

PythonGenStream::Line::Line(PythonGenStream &s)
    : s{s}, newline{true} {
  s.os.indent(s.indent);
}

PythonGenStream::Line::~Line() {
  if (newline)
    s.os << "\n";
}

PythonGenStream::Line::Line(Line &&line)
    : s{line.s}, newline{line.newline} {
  line.newline = false;
}

PythonGenStream::PythonGenStream(raw_ostream &os)
    : os{os}, indent{0} {}

PythonGenStream::Line PythonGenStream::line() {
  return Line{*this};
}

PythonGenStream &PythonGenStream::block(StringRef ty, StringRef expr) {
  line() << ty << " " << expr << ":";
  incr();
  return *this;
}

PythonGenStream &PythonGenStream::endblock() {
  decr();
  return *this;
}

void PythonGenStream::changeIndent(int delta) {
  indent += delta;
  assert(indent >= 0 && "Negative indent");
}

} // end namespace py
} // end namespace dmc
