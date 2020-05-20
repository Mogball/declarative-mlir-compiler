#include "Context.h"
#include "Support.h"
#include "Exception.h"
#include "Identifier.h"

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

bool isUnknownLoc(Location loc) {
  return loc.isa<UnknownLoc>();
}

/// CallSiteLoc.
bool isCallSiteLoc(Location loc) {
  return loc.isa<CallSiteLoc>();
}

CallSiteLoc toCallSiteLoc(Location loc) {
  if (!isCallSiteLoc(loc))
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
  if (!isFileLineColLoc(loc))
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

/// FusedLoc.
Location getFusedLoc(std::vector<Location> locs) {
  return FusedLoc::get(locs, getMLIRContext());
}

bool isFusedLoc(Location loc) {
  return loc.isa<FusedLoc>();
}

std::vector<Location> getLocations(Location loc) {
  if (!isFusedLoc(loc))
    throw invalid_cast{
        "Location is not a FusedLoc. Check with `isFusedLoc`."};
  return loc.cast<FusedLoc>().getLocations();
}

/// NameLoc.
Location getNameLoc(std::string name, Location child) {
  return NameLoc::get(getIdentifierChecked(name), child);
}

Location getNameLoc(std::string name) {
  return NameLoc::get(getIdentifierChecked(name), getMLIRContext());
}

bool isNameLoc(Location loc) {
  return loc.isa<NameLoc>();
}

NameLoc toNameLoc(Location loc) {
  if (!isNameLoc(loc))
    throw invalid_cast{
        "Location is not a NameLoc. Check with `isNameLoc`."};
  return loc.cast<NameLoc>();
}

std::string getName(Location loc) {
  return toNameLoc(loc).getName().str();
}

Location getChildLoc(Location loc) {
  return toNameLoc(loc).getChildLoc();
}

} // end namespace py
} // end namespace mlir
