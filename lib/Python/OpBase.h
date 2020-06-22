#pragma once

#include <mlir/IR/Operation.h>

class OpBase {
public:
  OpBase(mlir::Operation *op) : impl{op} {}
  operator mlir::Operation *() const { return impl; }
  operator mlir::Operation &() { return *impl; }
  operator bool() const { return impl; }

  template <typename OpTy> bool isa() const { return llvm::isa<OpTy>(impl); }

private:
  mlir::Operation *impl;
};
