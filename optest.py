from mlir import *
m = parseSourceFile('lua/dialect.mlir')
dialects = registerDynamicDialects(m)
lua = dialects[0]
luac = dialects[1]

m = ModuleOp("luaModule")

f = lua.func(name=StringAttr("luaFunc"),
             funcTy=TypeAttr(FunctionType([I64Type()], [I64Type()])))
m.append(f)

entry = Block()
f.body().append(entry)
entry.addArgs(f.funcTy().type.inputs)
b = Builder.atStart(entry)
b.create(lua.get_string, value=StringAttr("a_string_value"))
wrap = b.create(luac.wrap_int, val=entry.getArgument(0))
b.create(luac.unwrap_int, val=wrap.res())
b.create(lua.ret, arg=[wrap.res()])
print(m)
