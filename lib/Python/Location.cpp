#include "Context.h"
#include "Support.h"
#include "Identifier.h"

#include <mlir/IR/Location.h>

namespace mlir {
namespace py {

bool isUnknownLoc(Location loc) { return loc.isa<UnknownLoc>(); }
bool isCallSiteLoc(Location loc) { return loc.isa<CallSiteLoc>(); }
bool isFileLineColLoc(Location loc) { return loc.isa<FileLineColLoc>(); }
bool isFusedLoc(Location loc) { return loc.isa<FusedLoc>(); }
bool isNameLoc(Location loc) { return loc.isa<NameLoc>(); }

/// UnknownLoc.
UnknownLoc getUnknownLoc() {
  return UnknownLoc::get(getMLIRContext()).cast<UnknownLoc>();
}

/// CallSiteLoc.
CallSiteLoc getCallSiteLoc(Location callee, Location caller) {
  return CallSiteLoc::get(callee, caller).cast<CallSiteLoc>();
}

CallSiteLoc toCallSiteLoc(Location loc) {
  if (!isCallSiteLoc(loc))
    throw std::invalid_argument{
        "Location is not a CallSiteLoc. Check with `isCallSiteLoc`."};
  return loc.cast<CallSiteLoc>();
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

FileLineColLoc toFileLineColLoc(Location loc) {
  if (!isFileLineColLoc(loc))
    throw std::invalid_argument{
        "Location is not a FileLineColLoc. Check with `isFileLineColLoc`."};
  return loc.cast<FileLineColLoc>();
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

NameLoc toNameLoc(Location loc) {
  if (!isNameLoc(loc))
    throw std::invalid_argument{
        "Location is not a NameLoc. Check with `isNameLoc`."};
  return loc.cast<NameLoc>();
}

std::string getName(NameLoc loc) {
  return loc.getName().str();
}

Location getChildLoc(NameLoc loc) {
  return loc.getChildLoc();
}

} // end namespace py
} // end namespace mlir
