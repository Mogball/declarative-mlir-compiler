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
void exposeLocation(pybind11::module &m, pybind11::class_<Attribute> &attr);
void exposeArrayAttr(pybind11::module &m, pybind11::class_<Attribute> &attr);
void exposeDictAttr(pybind11::module &m, pybind11::class_<Attribute> &attr);
void exposeIntFPAttr(pybind11::module &m, pybind11::class_<Attribute> &attr);
void exposeSymbolRefAttr(pybind11::module &m, pybind11::class_<Attribute> &attr);
void exposeElementsAttr(pybind11::module &m, pybind11::class_<Attribute> &attr);

} // end namespace py
} // end namespace mlir
