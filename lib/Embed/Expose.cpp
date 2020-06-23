#include "Scope.h"
#include "dmc/Embed/InMemoryDef.h"
#include "dmc/Dynamic/DynamicType.h"
#include "dmc/Dynamic/DynamicOperation.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Traits/SpecTraits.h"

#include <llvm/ADT/STLExtras.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

using namespace pybind11;
using namespace llvm;
using namespace mlir;

namespace dmc {
namespace py {
namespace {

std::function<void(PythonGenStream::Line &)>
paramArgs(NamedParameterRange params) {
  return [params](PythonGenStream::Line &line) {
    interleaveComma(params, line, [&](auto param) {
      line << param.getName();
    });
  };
}

// TODO wish there was a Python API I could directly call, but one is not
// provided by pybind11, so I resort to codegen.
void exposeDynamicType(module &m, DynamicTypeImpl *impl) {
  InMemoryClass cls{impl->getName(), {"mlir.DynamicType"}, m};
  auto &s = cls.stream();
  s.line() << "def __init__(self, " << paramArgs(impl->getParamSpec())
      << "):" << incr; {
    s.line() << "super().__init__(mlir.build_dynamic_type(\""
        << impl->getDialect()->getNamespace() << "\", \"" << impl->getName()
        << "\", [" << paramArgs(impl->getParamSpec()) << "]))";
  } s.enddef();
  for (auto param : impl->getParamSpec()) {
    s.line() << "def " << param.getName() << "(self):" << incr; {
      s.line() << "return super().getParam(\"" << param.getName() << "\")";
    } s.enddef();
  }
}

struct Argument {
  Twine name;
  StringRef type;
};

struct ArgumentBuilder {
  void addArg(StringRef name, StringRef type) {
    args.push_back({Twine{name}, type});
  }

  void addDefaultArg(StringRef name, StringRef value, StringRef type) {
    defArgs.push_back({name + "=" + value, type});
  }

  SmallVector<Argument, 8> args;
  SmallVector<Argument, 8> defArgs;
};

void exposeDynamicOp(module &m, DynamicOperation *impl) {
  auto opName = impl->getName();
  InMemoryClass cls{opName.substr(opName.find('.') + 1),
                    {"mlir.Op", "mlir.OperationWrap"}, m};

  auto &s = cls.stream();
  auto opType = impl->getTrait<TypeConstraintTrait>()->getOpType();
  auto opAttr = impl->getTrait<AttrConstraintTrait>()->getOpAttrs();
  auto opSucc = impl->getTrait<SuccessorConstraintTrait>()->getOpSuccessors();
  auto opRegion = impl->getTrait<RegionConstraintTrait>()->getOpRegions();

  ArgumentBuilder b;
  b.addDefaultArg("loc", "mlir.UnknownLoc()", "mlir.Location");

  auto allArgs = concat<Argument>(b.args, b.defArgs);
  auto args = make_range(std::begin(allArgs), std::end(allArgs));
  {
    auto line = s.line() << "def __init__(self, ";
    interleaveComma(args, line, [&](const Argument &a) { line << a.name; });
    line << "):" << incr;
  }
  for (auto &a : args)
    s.line() << "assert(isinstance(" << a.name << ", " << a.type << "))";
  {
    auto line = s.line() << "theOp = Operation(loc, \"" << opName << "\", [";
    interleaveComma(opType.getResults(), line,
                    [&](const NamedType &ty) { line << ty.name << "Type"; });
    line << "], [";
    interleaveComma(opType.getOperands(), line,
                    [&](const NamedType &ty) { line << ty.name; });
    line << "], {";
    interleaveComma(opAttr, line, [&](const NamedAttribute &attr) {
      auto name = attr.first.strref();
      line << '"' << name << "\": " << name;
    });
    line << "}, [";
    interleaveComma(opSucc.getSuccessors(), line,
                    [&](const NamedConstraint &succ) { line << succ.name; });
    line << "], " << opRegion.getNumRegions() << ")";
  }
  s.line() << "Op.__init__(self, theOp)";
  s.line() << "OperationWrap.__init__(self, theOp)";
  s.enddef();

  llvm::errs() << cls.str() << "\n";
}

} // end anonymous namespace

module exposeDialect(DynamicDialect *dialect) {
  auto m = reinterpret_borrow<module>(
      PyImport_AddModule(dialect->getNamespace().str().c_str()));
  ensureBuiltins(m);
  exec("import mlir", m.attr("__dict__"));
  for (auto *ty : dialect->getTypes())
    exposeDynamicType(m, ty);
  for (auto *op : dialect->getOps())
    exposeDynamicOp(m, op);
  return m;
}

void exposeDialectInternal(DynamicDialect *dialect) {
  auto m = exposeDialect(dialect);
  auto regFcn = "register_internal_module_" + dialect->getNamespace().str();
  getInternalModule().def(regFcn.c_str(), [m]() { return m; });
  exec(dialect->getNamespace().str() + " = " + regFcn + "()",
       getInternalScope());
}

} // end namespace py
} // end namespace dmc
