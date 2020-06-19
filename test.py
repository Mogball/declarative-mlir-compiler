from mlir import *
module = parseSourceFile('spec/stencil.mlir')
dialects = registerDynamicDialects(module)
stencil = dialects[0]

shape = I64ArrayAttr([4, 5, 6])
f64Attr = TypeAttr(F64Type())

field = stencil.field(shape, f64Attr)
print("type:", field)
print("shape:", field.shape())
print("fieldTy:", field.type())

module = parseSourceFile('lua/dialect.mlir')
dialects = registerDynamicDialects(module)
lua = dialects[0]

print("lua value:", lua.value())
