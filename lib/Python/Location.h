#pragma once

#include <mlir/IR/Location.h>

namespace mlir {

std::ostream &operator<<(std::ostream &os, Location loc);

namespace py {

/// UnknownLoc.
Location getUnknownLoc();

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

} // end namespace py
} // end namespace mlir
