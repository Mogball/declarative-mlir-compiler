from mlir import *
m = parseSourceFile('lua/dialect.mlir')
dialects = registerDynamicDialects(m)
lua = dialects[0]

op = lua.get_string(value=StringAttr("a_string_value"))
print(op.getNumRegions())
print(op.value())
print(op.res())
