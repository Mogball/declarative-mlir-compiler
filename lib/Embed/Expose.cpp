#include "Scope.h"
#include "dmc/Embed/InMemoryDef.h"
#include "dmc/Dynamic/Alias.h"
#include "dmc/Dynamic/DynamicType.h"
#include "dmc/Dynamic/DynamicOperation.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Traits/SpecTraits.h"
#include "dmc/Spec/SpecAttrs.h"
#include "dmc/Spec/SpecTypes.h"
#include "dmc/Spec/SpecRegion.h"
#include "dmc/Spec/SpecSuccessor.h"
#include "dmc/Python/Polymorphic.h"

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
  const StringRef name, type, value;
};

struct ArgumentBuilder {
  void add(StringRef name, StringRef type) {
    args.push_back({name, type, "None"});
  }
  void add(StringRef name, StringRef value, StringRef type) {
    defArgs.push_back({name, type, value});
  }

  SmallVector<Argument, 8> args;
  SmallVector<Argument, 8> defArgs;
};

TypeMetadata *lookupTypeData(DynamicContext *ctx, Type ty) {
  if (auto *dialect = ctx->lookupDialectFor(ty))
    return dialect->lookupTypeData(ty);
  return nullptr;
}

AttributeMetadata *lookupAttributeData(DynamicContext *ctx, Attribute attr) {
  if (auto *dialect = ctx->lookupDialectFor(attr))
    return dialect->lookupAttributeData(attr);
  return nullptr;
}

template <typename T> auto inlineOrConcat(PythonGenStream::Line &line) {
  return [&](const T &t) {
    line << "] + " << (t.isVariadic() ? "" : "[") << t.name
         << (t.isVariadic() ? "" : "]") << " + [";
  };
}

void exposeDynamicOp(module &m, DynamicOperation *impl) {
  auto *dialect = impl->getDialect();
  auto *ctx = dialect->getDynContext();

  // Declare the class
  auto opName = impl->getName();
  InMemoryClass cls{opName.substr(opName.find('.') + 1),
                    {"mlir.Op", "mlir.OperationWrap"}, m};

  // Retrieve op traits
  auto &s = cls.stream();
  auto opType = impl->getTrait<TypeConstraintTrait>()->getOpType();
  auto opAttr = impl->getTrait<AttrConstraintTrait>()->getOpAttrs();
  auto opSucc = impl->getTrait<SuccessorConstraintTrait>()->getOpSuccessors();
  auto opRegion = impl->getTrait<RegionConstraintTrait>()->getOpRegions();

  // Collect arguments, checking for buildable types and attributes
  ArgumentBuilder b;
  for (auto &[name, type] : opType.getOperands()) {
    b.add(name, type.isa<VariadicType>() ? "list" : "mlir.Value");
  }
  for (auto &[name, type] : opType.getResults()) {
    auto cls = type.isa<VariadicType>() ? "list" : "mlir.Type";
    if (auto *data = lookupTypeData(ctx, type); data && data->getBuilder()) {
      b.add(name, *data->getBuilder(), cls);
    } else {
      b.add(name, cls);
    }
  }
  for (auto &[name, attr] : opAttr) {
    if (auto *data = lookupAttributeData(ctx, attr); data && data->getBuilder()) {
      b.add(name, *data->getBuilder(), "mlir.Attribute");
    } else if (attr.isa<DefaultAttr>()) {
      throw std::runtime_error{"default attributes expose not implemented"};
    } else if (attr.isa<OptionalAttr>()) {
      b.add(name, "mlir.Attribute()", "mlir.Attribute");
    } else {
      b.add(name, "mlir.Attribute");
    }
  }
  for (auto &[name, succ] : opSucc.getSuccessors()) {
    b.add(name, succ.isa<VariadicSuccessor>() ? "list" : "mlir.Block");
  }
  auto numRegionsStr = std::to_string(opRegion.size()); // keep on stack
  b.add("numRegions", numRegionsStr, "int");
  b.add("loc", "mlir.UnknownLoc()", "mlir.LocationAttr");

  // Declare the constructor
  auto allArgs = concat<Argument>(b.args, b.defArgs);
  auto args = make_range(std::begin(allArgs), std::end(allArgs));
  {
    auto line = s.line() << "def __init__(self, theOp=None, ";
    interleaveComma(args, line, [&](const Argument &a) {
      line << a.name << "=" << a.value;
    });
    line << "):" << incr;
  }

  s.if_("not theOp");

  // Insert type checks
  for (auto &a : b.args) {
    s.line() << "assert " << a.name << " != None, \"missing positional argument"
        << " '" << a.name << "'\"";
  }
  for (auto &a : args) {
    s.line() << "assert isinstance(" << a.name << ", " << a.type << "), "
      << "\"expected '" << a.name << "' to be of type '" << a.type
      << "' but got \" + str(type(" << a.name << "))";
  }

  // Call to generic operation constructor
  {
    s.line() << "theAttrs = {}";
    for (auto &attr : opAttr) {
      auto name = attr.first.strref();
      s.if_(name); {
        s.line() << "theAttrs[\"" << name << "\"] = " << name;
      } s.endif();
    }
  }
  {
    auto line = s.line() << "theOp = mlir.Operation(loc, \"" << opName << "\", [";
    interleave(opType.getResults(), line,
               inlineOrConcat<NamedType>(line), "");
    line << "], [";
    interleave(opType.getOperands(), line,
               inlineOrConcat<NamedType>(line), "");
    line << "], theAttrs, [";
    interleave(opSucc.getSuccessors(), line,
               inlineOrConcat<NamedConstraint>(line), "");
    line << "], numRegions)";
  }

  s.endif();

  // Call to parent class constructors
  s.line() << "mlir.Op.__init__(self, theOp)";
  s.line() << "mlir.OperationWrap.__init__(self, theOp)";
  s.enddef();

  // Define getters
  s.line() << "@staticmethod";
  s.def("getName()"); {
    s.line() << "return \"" << impl->getName() << "\"";
  } s.enddef();
  for (auto &[name, type] : opType.getOperands()) {
    auto getter = type.isa<VariadicType>() ? "getOperandGroup" : "getOperand";
    s.def(name + "(self)"); {
      s.line() << "return mlir.OperationWrap." << getter << "(self, \"" << name
        << "\")";
    } s.enddef();
  }
  for (auto &[name, type] : opType.getResults()) {
    auto getter = type.isa<VariadicType>() ? "getResultGroup" : "getResult";
    s.def(name + "(self)"); {
      s.line() << "return mlir.OperationWrap." << getter << "(self, \"" << name
          << "\")";
    } s.enddef();
  }
  for (auto &[name, attr] : opAttr) {
    s.def(name + "(self)"); {
      s.line() << "return mlir.Op.getAttr(self, \"" << name << "\")";
    } s.enddef();
  }
  for (auto &[name, succ] : opSucc.getSuccessors()) {
    auto getter = succ.isa<VariadicSuccessor>() ?
        "getSuccessors" : "getSuccessor";
    s.def(name + "(self)"); {
      s.line() << "return mlir.OperationWrap." << getter << "(self, \"" << name
          << "\")";
    } s.enddef();
  }
  for (auto &[name, region] : opRegion.getRegions()) {
    auto getter = region.isa<VariadicRegion>() ? "getRegions" : "getRegion";
    s.def(name + "(self)"); {
      s.line() << "return mlir.OperationWrap." << getter << "(self, \"" << name
          << "\")";
    } s.enddef();
  }
}

} // end anonymous namespace

void exposeDialectInternal(DynamicDialect *dialect, ArrayRef<StringRef> scope) {
  auto m = reinterpret_borrow<module>(
      PyImport_AddModule(dialect->getNamespace().str().c_str()));
  ensureBuiltins(m);
  exec("import mlir", m.attr("__dict__"));
  exec("from mlir import *", m.attr("__dict__"));
  auto regFcn = "register_internal_module_" + dialect->getNamespace().str();
  getInternalModule().def(regFcn.c_str(), [m]() { return m; });
  exec(dialect->getNamespace().str() + " = " + regFcn + "()",
       getInternalScope());
  exec("name = \"" + dialect->getNamespace().str() + "\"",
       m.attr("__dict__"));
  for (auto name : scope) {
    exec(name.str() + " = mlir.register_internal_module_" +
         name.str() + "()", m.attr("__dict__"));
  }
  for (auto *ty : dialect->getTypes()) {
    exposeDynamicType(m, ty);
  }
  for (auto *op : dialect->getOps()) {
    exposeDynamicOp(m, op);
  }
  for (auto *ty : dialect->getTypeAliases()) {
    auto name = ty->getName().str();
    m.def(name.c_str(), [ty]() { return ty->getAliasedType(); });
  }
  for (auto *attr : dialect->getAttrAliases()) {
    auto name = attr->getName().str();
    m.def(name.c_str(), [attr]() { return attr->getAliasedAttr(); });
  }
}

} // end namespace py
} // end namespace dmc
