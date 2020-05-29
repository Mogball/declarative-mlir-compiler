#include "Context.h"
#include "Utility.h"
#include "Identifier.h"

#include <mlir/IR/Location.h>

namespace mlir {
namespace py {

/// UnknownLoc.
UnknownLoc getUnknownLoc() {
  return UnknownLoc::get(getMLIRContext()).cast<UnknownLoc>();
}

/// CallSiteLoc.
CallSiteLoc getCallSiteLoc(Location callee, Location caller) {
  return CallSiteLoc::get(callee, caller).cast<CallSiteLoc>();
}

Location getCallee(CallSiteLoc loc) {
  return loc.getCallee();
}

Location getCaller(CallSiteLoc loc) {
  return loc.getCaller();
}

/// FileLineColLoc.
FileLineColLoc getFileLineColLoc(std::string filename, unsigned line,
                                 unsigned col) {
  return FileLineColLoc::get(filename, line, col, getMLIRContext())
      .cast<FileLineColLoc>();
}

std::string getFilename(FileLineColLoc loc) {
  return loc.getFilename().str();
}

unsigned getLine(FileLineColLoc loc) {
  return loc.getLine();
}

unsigned getColumn(FileLineColLoc loc) {
  return loc.getColumn();
}

/// FusedLoc.
FusedLoc getFusedLoc(const std::vector<Location> &locs) {
  return FusedLoc::get(locs, getMLIRContext()).cast<FusedLoc>();
}

std::vector<Location> *getLocations(FusedLoc loc) {
  return new std::vector<Location>{loc.getLocations()};
}

/// NameLoc.
NameLoc getNameLoc(std::string name, Location child) {
  return NameLoc::get(getIdentifierChecked(name), child).cast<NameLoc>();
}

NameLoc getNameLoc(std::string name) {
  return NameLoc::get(getIdentifierChecked(name), getMLIRContext())
      .cast<NameLoc>();
}

std::string getName(NameLoc loc) {
  return loc.getName().str();
}

Location getChildLoc(NameLoc loc) {
  return loc.getChildLoc();
}

} // end namespace py
} // end namespace mlir
