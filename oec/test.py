import inspect
import ast
import astpretty
import numpy
import os
from mlir import *

################################################################################
# Initialization
################################################################################

cwd = os.path.dirname(os.path.realpath(__file__))

def get_dialects(filename=cwd + '/python.mlir'):
    m = parseSourceFile(filename)
    assert m, "failed to load dialects"
    dialects = registerDynamicDialects(m)
    assert dialects, "failed to register dialects"
    return dialects[0]

py = get_dialects()

################################################################################
# AST Visitor
################################################################################

def nonnull_object(v):
    return isinstance(v, Value) and v.type == py.object()

def nonnull_handle(v):
    return isinstance(v, Value) and v.type == py.handle()

def load_ctx(n):
    return isinstance(n.ctx, ast.Load)

class StencilProgramVisitor(ast.NodeVisitor):

    def __init__(self):
        self.b = Builder()
        self.filename = "stencil.program"

    def start_loc(self, node):
        return FileLineColLoc(self.filename, node.lineno, node.col_offset)

    def end_loc(self, node):
        return FileLineColLoc(self.filename, node.end_lineno,
                              node.end_col_offset)

    def generic_visit(self, node):
        return ast.NodeVisitor.generic_visit(self, node)


    def visit_Module(self, node):
        self.m = ModuleOp()
        self.b.insertAtStart(self.m.getRegion(0).getBlock(0))
        for funcDef in node.body:
            assert isinstance(funcDef, ast.FunctionDef)
            self.visit(funcDef)
        return self.m

    def visit_FunctionDef(self, node):
        assert isinstance(node.args, ast.arguments)
        arg_names, tys, locs = self.visit(node.args)
        rets = [] if node.returns == None else [py.object()]
        func = self.b.create(py.func, name=StringAttr(node.name),
                             sig=TypeAttr(FunctionType(tys, rets)),
                             loc=self.start_loc(node))
        b = func.body().addEntryBlock(tys)
        ip = self.b.saveIp()
        self.b.insertAtStart(b)
        for name, arg, loc in zip(arg_names, b.getArguments(), locs):
            ref = self.b.create(py.name, var=StringAttr(name), loc=loc).ref()
            self.b.create(py.store, ref=ref, arg=arg, loc=loc)
        for stmt in node.body:
            self.visit(stmt)
        self.b.restoreIp(ip)

    def visit_arguments(self, node):
        assert len(node.posonlyargs) == 0
        assert node.vararg == None
        assert len(node.kwonlyargs) == 0
        assert len(node.kw_defaults) == 0
        assert node.kwarg == None
        assert len(node.defaults) == 0
        names = []
        tys = []
        locs = []
        for arg in node.args:
            assert isinstance(arg, ast.arg)
            name, ty = self.visit(arg)
            names.append(name)
            tys.append(ty)
            locs.append(self.start_loc(arg))
        return names, tys, locs

    def visit_arg(self, node):
        assert node.type_comment == None
        # TODO check type annotation
        return node.arg, py.object()

    def visit_Assign(self, node):
        assert len(node.targets) == 1, "multiple assign unimplemented"
        ref = self.visit(node.targets[0])
        assert nonnull_handle(ref)
        arg = self.visit(node.value)
        assert nonnull_object(arg)
        self.b.create(py.store, ref=ref, arg=arg, loc=self.start_loc(node))

    def visit_Expr(self, node):
        self.visit(node.value)

    def visit_Call(self, node):
        func = self.visit(node.func)
        assert nonnull_object(func)
        args = [self.visit(arg) for arg in node.args]
        assert all(nonnull_object(arg) for arg in args)
        return self.b.create(py.call, func=func, args=args,
                             loc=self.start_loc(node)).rets()

    def visit_Return(self, node):
        args = [] if node.value == None else [self.visit(node.value)]
        assert all(nonnull_object(arg) for arg in args)
        self.b.create(py.ret, args=args, loc=self.start_loc(node))

    def visit_Name(self, node):
        ref = self.b.create(py.name, var=StringAttr(node.id),
                            loc=self.start_loc(node)).ref()
        return ref if isinstance(node.ctx, ast.Store) else \
            self.b.create(py.load, ref=ref, loc=self.start_loc(node)).res()

    def visit_Load(self, node):
        pass

    def visit_Store(self, node):
        pass

    def visit_Attribute(self, node):
        arg = self.visit(node.value)
        assert load_ctx(node), "store to attribute unimplemented"
        assert nonnull_object(arg)
        return self.b.create(py.attribute, arg=arg, name=StringAttr(node.attr),
                             loc=self.start_loc(node)).res()

    def visit_Subscript(self, node):
        assert load_ctx(node), "store to subscript unimplemented"
        arg, idx = self.visit(node.value), self.visit(node.slice)
        assert nonnull_object(arg) and nonnull_object(idx)
        return self.b.create(py.subscript, arg=arg, idx=idx,
                             loc=self.start_loc(node)).res()

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Constant(self, node):
        if isinstance(node.value, int):
            value = I64Attr(node.value)
        else:
            raise NotImplementedError("constant type: " + str(type(node.value)))
        return self.b.create(py.constant, value=value,
                             loc=self.start_loc(node)).res()

    def visit_Tuple(self, node):
        assert load_ctx(node), "store to tuple unimplemented"
        elts = [self.visit(el) for el in node.elts]
        assert all(nonnull_object(el) for el in elts)
        return self.b.create(py.make_tuple, elts=elts,
                             loc=self.start_loc(node)).res()

    def visit_List(self, node):
        assert load_ctx(node), "store to list unimplemented"
        elts = [self.visit(el) for el in node.elts]
        assert all(nonnull_object(el) for el in elts)
        return self.b.create(py.make_list, elts=elts,
                             loc=self.start_loc(node)).res()

    def visit_UnaryOp(self, node):
        arg = self.visit(node.operand)
        assert nonnull_object(arg)
        return self.b.create(py.unary, arg=arg,
                             op=StringAttr(self.visit(node.op)),
                             loc=self.start_loc(node)).res()

    def visit_USub(self, node):
        return "-"

    def visit_BinOp(self, node):
        lhs, rhs = self.visit(node.left), self.visit(node.right)
        assert nonnull_object(lhs) and nonnull_object(rhs)
        return self.b.create(py.binary, lhs=lhs, rhs=rhs,
                             op=StringAttr(self.visit(node.op)),
                             loc=self.start_loc(node)).res()

    def visit_Add(self, node):
        return "+"

    def visit_Mult(self, node):
        return "*"

################################################################################
# Test Area
################################################################################

stencil = None

def laplace(a:numpy.ndarray, b:numpy.ndarray):
  stencil.cast(a, [-4, -4, -4], [68, 68, 68])
  stencil.cast(b, [-4, -4, -4], [68, 68, 68])
  atmp = stencil.load(a)

  def applyFcn(c):
    return -4 * c[0, 0, 0] + c[-1, 0, 0] + c[1, 0, 0] + c[0, 1, 0] + c[0, -1, 0]

  btmp = stencil.apply(atmp, applyFcn)
  stencil.store(b, btmp, [0, 0, 0], [64, 64, 64])
  return

node = ast.parse(inspect.getsource(laplace))
visitor = StencilProgramVisitor()
m = visitor.visit(node)
#astpretty.pprint(node)
print(m)
verify(m)
