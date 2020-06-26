#include "Type.h"
#include "Context.h"
#include "Expose.h"
#include "Utility.h"
#include "dmc/Dynamic/DynamicType.h"

#include <mlir/IR/Dialect.h>
#include <mlir/IR/StandardTypes.h>

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

using namespace pybind11;
using namespace llvm;
using namespace mlir;

namespace mlir {
namespace py {

template <typename FcnT> auto nullcheck(FcnT fcn) {
  return ::nullcheck(fcn, "type");
}

TypeClass exposeTypeBase(module &m) {
  class_<Type> type{m, "Type"};
  type
      .def(init<>())
      .def(init<Type>())
      .def(self == self)
      .def(self != self)
      .def("__eq__", &Type::operator==)
      .def("__ne__", &Type::operator!=)
      .def("__repr__", StringPrinter<Type>{})
      .def("__bool__", &Type::operator bool)
      .def("__invert__", &Type::operator!)
      .def("__hash__", overload<hash_code(Type)>(&hash_value))
      .def_property_readonly("kind", nullcheck(&Type::getKind))
      .def_property_readonly("dialect", nullcheck(&Type::getDialect),
                             return_value_policy::reference)
      .def("isIndex", nullcheck(&Type::isIndex))
      .def("isBF16", nullcheck(&Type::isBF16))
      .def("isF16", nullcheck(&Type::isF16))
      .def("isF32", nullcheck(&Type::isF32))
      .def("isF64", nullcheck(&Type::isF64))
      .def("isInteger", nullcheck(&Type::isInteger))
      .def("isSignlessInteger", nullcheck(
          overload<bool(Type::*)()>(&Type::isSignlessInteger)))
      .def("isSignlessInteger", nullcheck(
          overload<bool(Type::*)(unsigned)>(&Type::isSignlessInteger)))
      .def("isSignedInteger", nullcheck(
          overload<bool(Type::*)()>(&Type::isSignedInteger)))
      .def("isSignedInteger", nullcheck(
          overload<bool(Type::*)(unsigned)>(&Type::isSignedInteger)))
      .def("isUnsignedInteger", nullcheck(
          overload<bool(Type::*)()>(&Type::isSignedInteger)))
      .def("isUnsignedInteger", nullcheck(
          overload<bool(Type::*)(unsigned)>(&Type::isSignedInteger)))
      .def("getIntOrFloatBitWidth", nullcheck(&getIntOrFloatBitWidth))
      .def("isSignlessIntOrIndex", nullcheck(&Type::isSignlessIntOrIndex))
      .def("isSignlessIntOrIndexOrFloat",
           nullcheck(&Type::isSignlessIntOrIndexOrFloat))
      .def("isSignlessIntOrFloat", nullcheck(&Type::isSignlessIntOrFloat))
      .def("isIntOrIndex", nullcheck(&Type::isIntOrIndex))
      .def("isIntOrFloat", nullcheck(&Type::isIntOrFloat))
      .def("isIntOrIndexOrFloat", nullcheck(&Type::isIntOrIndexOrFloat));

  return type;
}

void exposeType(module &m, TypeClass &type) {
  exposeFunctionType(m, type);
  exposeOpaqueType(m, type);
  exposeStandardNumericTypes(m, type);
  exposeShapedTypes(m, type);

  class_<TupleType>(m, "TupleType", type)
      .def(init([](TypeListRef elTys) {
        return TupleType::get(elTys, getMLIRContext());
      }), "elementTypes"_a = TypeList{})
      .def_property_readonly("types", nullcheck([](TupleType ty) {
        auto elTys = ty.getTypes();
        return new TypeList{std::begin(elTys), std::end(elTys)};
      }))
      .def("__len__", nullcheck(&TupleType::size))
      .def("__iter__", nullcheck([](TupleType ty) {
        return make_iterator(ty.begin(), ty.end());
      }), keep_alive<0, 1>())
      .def("__getitem__", nullcheck([](TupleType ty, size_t idx) {
        if (idx >= ty.size())
          throw index_error{};
        return ty.getType(idx);
      }))
      .def("getFlattenedTypes", nullcheck([](TupleType ty) {
        SmallVector<Type, 8> elTys;
        ty.getFlattenedTypes(elTys);
        return new TypeList{std::begin(elTys), std::end(elTys)};
      }));

  using dmc::DynamicType;

  class_<DynamicType>(m, "DynamicType", type)
      .def(init<DynamicType>())
      .def("getParams", [](DynamicType ty) {
        auto params = ty.getParams();
        std::vector<Attribute> ret{std::begin(params), std::end(params)};
        return ret;
      })
      .def("getParam", [](DynamicType ty, std::string name) {
        return ty.getParam(name);
      });

  implicitly_convertible_from_all<Type,
      FunctionType, OpaqueType,
      ComplexType, IndexType, IntegerType, FloatType, mlir::NoneType,

      VectorType,
      TensorType, RankedTensorType, UnrankedTensorType,
      BaseMemRefType, MemRefType, UnrankedMemRefType,

      TupleType, DynamicType>(type);
}

} // end namespace py
} // end namespace mlir
