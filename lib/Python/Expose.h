#pragma once

#include <mlir/IR/Attributes.h>
#include <pybind11/pybind11.h>

namespace mlir {
namespace py {

void exposeParser(pybind11::module &m);
void exposeModule(pybind11::module &m);
void exposeType(pybind11::module &m);
void exposeAttribute(pybind11::module &m);

/// Attribute subclasses.
using AttrClass = pybind11::class_<Attribute>;
void exposeLocation(pybind11::module &m, AttrClass &attr);
void exposeArrayAttr(pybind11::module &m, AttrClass &attr);
void exposeDictAttr(pybind11::module &m, AttrClass &attr);
void exposeIntFPAttr(pybind11::module &m, AttrClass &attr);
void exposeSymbolRefAttr(pybind11::module &m, AttrClass &attr);
void exposeElementsAttr(pybind11::module &m, AttrClass &attr);

/// Type subclasses.
using TypeClass = pybind11::class_<Type>;
void exposeFunctionType(pybind11::module &m, TypeClass &type);
void exposeOpaqueType(pybind11::module &m, TypeClass &type);
void exposeStandardNumericTypes(pybind11::module &m, TypeClass &type);
void exposeShapedTypes(pybind11::module &m, TypeClass &type);

} // end namespace py
} // end namespace mlir
