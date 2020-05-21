#pragma once

#include <mlir/IR/Location.h>

namespace mlir {
namespace py {

bool isUnknownLoc(Location loc);
bool isCallSiteLoc(Location loc);
bool isFileLineColLoc(Location loc);
bool isFusedLoc(Location loc);
bool isNameLoc(Location loc);

/// UnknownLoc.
UnknownLoc getUnknownLoc();

/// CallSiteLoc.
CallSiteLoc getCallSiteLoc(Location callee, Location caller);
Location getCallee(CallSiteLoc loc);
Location getCaller(CallSiteLoc loc);

/// FileLineColLoc.
FileLineColLoc getFileLineColLoc(std::string filename, unsigned line,
                                 unsigned col);
std::string getFilename(FileLineColLoc loc);
unsigned getLine(FileLineColLoc loc);
unsigned getColumn(FileLineColLoc loc);

/// FusedLoc.
FusedLoc getFusedLoc(const std::vector<Location> &locs);
std::vector<Location> *getLocations(FusedLoc loc);

/// NameLoc.
NameLoc getNameLoc(std::string name, Location child);
NameLoc getNameLoc(std::string name);
std::string getName(NameLoc loc);
Location getChildLoc(NameLoc loc);

} // end namespace py
} // end namespace mlir
