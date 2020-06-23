from mlir import *
m = parseSourceFile('lua/dialect.mlir')
dialects = registerDynamicDialects(m)
lua = dialects[0]

m = ModuleOp("luaModule")
f = FuncOp("luaFunc", FunctionType([I32Type(), I32Type()], [I32Type()]))
m.append(f)

entry = f.addEntryBlock()
b = Builder.atStart(entry)
b.create(lua.get_string, value=StringAttr("a_string_value"))
print(m)
