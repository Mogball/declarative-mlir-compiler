#include "Context.h"
#include "Support.h"
#include "Exception.h"

#include <mlir/IR/Location.h>

namespace mlir {

std::ostream &operator<<(std::ostream &os, Location loc) {
  return printToOs(os, loc);
}

namespace py {

/// UnknownLoc.
Location getUnknownLoc() {
  return UnknownLoc::get(getMLIRContext());
}

/// CallSiteLoc.
bool isCallSiteLoc(Location loc) {
  return loc.isa<CallSiteLoc>();
}

CallSiteLoc toCallSiteLoc(Location loc) {
  if (!loc.isa<CallSiteLoc>())
    throw invalid_cast{
        "Location is not a CallSiteLoc. Check with `isCallSiteLoc`."};
  return loc.cast<CallSiteLoc>();
}

Location getCallee(Location loc) {
  return toCallSiteLoc(loc).getCallee();
}

Location getCaller(Location loc) {
  return toCallSiteLoc(loc).getCaller();
}

/// FileLineColLoc.
Location getFileLineColLoc(std::string filename, unsigned line, unsigned col) {
  return FileLineColLoc::get(filename, line, col, getMLIRContext());
}

bool isFileLineColLoc(Location loc) {
  return loc.isa<FileLineColLoc>();
}

FileLineColLoc toFileLineColLoc(Location loc) {
  if (!loc.isa<FileLineColLoc>())
    throw invalid_cast{
        "Location is not a FileLineColLoc. Check with `isFileLineColLoc`."};
  return loc.cast<FileLineColLoc>();
}

std::string getFilename(Location loc) {
  return toFileLineColLoc(loc).getFilename().str();
}

unsigned getLine(Location loc) {
  return toFileLineColLoc(loc).getLine();
}

unsigned getColumn(Location loc) {
  return toFileLineColLoc(loc).getColumn();
}

} // end namespace py
} // end namespace mlir
