from mlir import *
m = parseSourceFile('lua/dialect.mlir')
dialects = registerDynamicDialects(m)
lua = dialects[0]

class get_string(OperationWrap, Op):
    def __init__(self, value, resType = lua.value(), loc = UnknownLoc()):
        op = Operation(loc, "lua.get_string", [resType], [], {"value": value}, [], 0)
        OperationWrap.__init__(self, op)
        Op.__init__(self, op)

op = get_string(value=StringAttr("a_string_value"))
print(op.getResult("res"))
print(op.getNumRegions())
