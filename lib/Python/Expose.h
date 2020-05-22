#pragma once

#include <mlir/IR/Attributes.h>
#include <pybind11/pybind11.h>

namespace mlir {
namespace py {

using AttrClass = pybind11::class_<Attribute>;
using TypeClass = pybind11::class_<Type>;

void exposeParser(pybind11::module &m);
void exposeModule(pybind11::module &m);
/// pybind11 needs Type to be exposed before it can be used in default args.
TypeClass exposeTypeBase(pybind11::module &m);
void exposeType(pybind11::module &m, TypeClass &type);
void exposeAttribute(pybind11::module &m);
void exposeOps(pybind11::module &m);

/// Attribute subclasses.
void exposeLocation(pybind11::module &m, AttrClass &attr);
void exposeArrayAttr(pybind11::module &m, AttrClass &attr);
void exposeDictAttr(pybind11::module &m, AttrClass &attr);
void exposeIntFPAttr(pybind11::module &m, AttrClass &attr);
void exposeSymbolRefAttr(pybind11::module &m, AttrClass &attr);
void exposeElementsAttr(pybind11::module &m, AttrClass &attr);

/// Type subclasses.
void exposeFunctionType(pybind11::module &m, TypeClass &type);
void exposeOpaqueType(pybind11::module &m, TypeClass &type);
void exposeStandardNumericTypes(pybind11::module &m, TypeClass &type);
void exposeShapedTypes(pybind11::module &m, TypeClass &type);

} // end namespace py
} // end namespace mlir
