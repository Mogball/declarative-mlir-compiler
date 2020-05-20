#pragma once

#include <mlir/IR/Location.h>

namespace mlir {
namespace py {

/// UnknownLoc.
Location getUnknownLoc();
bool isUnknownLoc(Location loc);

/// CallSiteLoc.
bool isCallSiteLoc(Location loc);
Location getCallee(Location loc);
Location getCaller(Location loc);

/// FileLineColLoc.
Location getFileLineColLoc(std::string filename, unsigned line, unsigned col);
bool isFileLineColLoc(Location loc);
std::string getFilename(Location loc);
unsigned getLine(Location loc);
unsigned getColumn(Location loc);

/// FusedLoc.
Location getFusedLoc(std::vector<Location> locs);
bool isFusedLoc(Location loc);
std::vector<Location> getLocations(Location loc);

/// NameLoc.
Location getNameLoc(std::string name, Location child);
Location getNameLoc(std::string name);
bool isNameLoc(Location loc);
std::string getName(Location loc);
Location getChildLoc(Location loc);

} // end namespace py
} // end namespace mlir
