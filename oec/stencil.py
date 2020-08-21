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

def get_dialects(filename):
    m = parseSourceFile(cwd + '/' + filename)
    assert m, "failed to load dialects"
    dialects = registerDynamicDialects(m)
    assert dialects, "failed to register dialects"
    return dialects

py, stencil, tmp = get_dialects("python.mlir")

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
            self.b.create(py.assign, ref=ref, arg=arg, loc=loc)
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
        self.b.create(py.assign, ref=ref, arg=arg, loc=self.start_loc(node))

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
# Variable Allocation
################################################################################

class VarAllocVisitor:
    def __init__(self):
        self.vars = {}
        self.b = Builder()

    def visit(self, func):
        if not isa(func, py.func):
            return
        func = py.func(func)
        ops = list(op for op in func.body().getBlock(0))
        for op in ops:
            self.visit_op(op)

    def visit_op(self, op):
        if isa(op, py.func):
            VarAllocVisitor().visit(op)
        elif isa(op, py.assign):
            op = py.assign(op)
            var = py.name(op.ref().definingOp).var().getValue()
            self.vars[var] = op.new()
        elif isa(op, py.name):
            op = py.name(op)
            var = op.var().getValue()
            if var in self.vars:
                self.b.replace(op, [self.vars[var]])

def elideLoad(op, b):
    if not isa(op.ref().definingOp, py.assign):
        return False
    assign = py.assign(op.ref().definingOp)
    b.replace(op, [assign.arg()])

def elideAssign(op, b):
    if not op.new().useEmpty() or op.arg().hasOneUse():
        return False
    b.erase(op)

def varAllocPass(m):
    for f in m:
        VarAllocVisitor().visit(f)
    applyCSE(m, lambda x: True)
    applyOptPatterns(m, [
        Pattern(py.load, elideLoad),
        Pattern(py.assign, elideAssign),
    ])

################################################################################
# Raise Pass
################################################################################

def raiseStencilModule(op, b):
    if not isa(op.ref().definingOp, py.name):
        return False
    name = py.name(op.ref().definingOp)
    if name.var().getValue() != "stencil":
        return False
    module = b.create(tmp.stencil_module, loc=op.loc).res()
    b.replace(op, [module])

def raiseStencilFcn(name, tmp_op):
    def patternFcn(op, b):
        if op.name().getValue() != name:
            return False
        fcn = b.create(tmp_op, loc=op.loc).res()
        b.replace(op, [fcn])
    return Pattern(py.attribute, patternFcn)

def evalConstExpr(op):
    if isa(op, py.constant):
        return py.constant(op).value().getInt()
    elif isa(op, py.unary):
        op = py.unary(op)
        assert op.op().getValue() == "-"
        return -evalConstExpr(op.arg().definingOp)
    elif isa(op, py.binary):
        op = py.binary(op)
        lhs = evalConstExpr(op.lhs().definingOp)
        rhs = evalConstExpr(op.rhs().definingOp)
        binOp = op.op().getValue()
        if binOp == "+":
            return lhs + rhs
        elif binOp == "-":
            return lhs - rhs
        elif binOp == "*":
            return lhs * rhs;
        elif binOp == "/":
            return lhs / rhs
        else:
            raise NotImplementedError("unrecognized constexpr binary op: " +
                                      binOp)
    else:
        raise NotImplementedError("unrecognized constexpr operation: " +
                                  str(op))

def raiseTupleOrListToIndex(op, b):
    vals = list(evalConstExpr(el.definingOp) for el in op.elts())
    index = b.create(tmp.stencil_index, index=I64ArrayAttr(vals),
                     loc=op.loc).res()
    b.replace(op, [index])

def raiseFunctionRef(op, b):
    if not isa(op.ref().definingOp, py.name):
        return False
    name = py.name(op.ref().definingOp)
    if not isa(op.parentOp, py.func):
        return False
    func = py.func(op.parentOp)
    for o in func.body().getBlock(0):
        if isa(o, py.func) and py.func(o).name() == name.var():
            f = py.func(o)
            apply = b.create(tmp.stencil_apply_body, loc=op.loc)
            f.body().cloneInto(apply.body())
            b.erase(o)
            b.replace(op, [apply.res()])
            return
    return False

def get_index(arg):
    assert isa(arg.definingOp, tmp.stencil_index)
    return tmp.stencil_index(arg.definingOp).index()

def convert_stencil_sig(sig):
    args = [tmp.unshaped_f64_field() for field in sig.inputs]
    rets = [tmp.unshaped_f64_field() for field in sig.results]
    return FunctionType(args, rets)

def copy_into(newBlk, oldBlk, bvm):
    for oldOp in oldBlk:
        newBlk.append(oldOp.clone(bvm))

def raiseStencilProgram(op, b):
    if not op.parentOp or not isa(op.parentOp, ModuleOp):
        return False
    func = b.create(FuncOp, name=op.name().getValue(),
                    type=convert_stencil_sig(op.sig().type),
                    attrs={"stencil.program":UnitAttr()})
    bvm = BlockAndValueMapping()
    entry = func.addEntryBlock()
    body = op.body().getBlock(0)
    for field, obj in zip(entry.getArguments(), body.getArguments()):
        bvm[obj] = field
    copy_into(entry, body, bvm)
    b.erase(op)
    return True

def raiseStencilAssert(op, b):
    if not isa(op.func().definingOp, tmp.stencil_assert):
        return False
    assert op.rets().useEmpty()
    args = op.args()
    b.create(stencil.Assert, field=args[0], lb=get_index(args[1]),
             ub=get_index(args[2]), loc=op.loc)
    b.erase(op)

def raiseStencilLoad(op, b):
    if not isa(op.func().definingOp, tmp.stencil_load):
        return False
    args = op.args()
    lb = Attribute() if len(args) == 1 else get_index(args[1])
    ub = Attribute() if len(args) == 1 else get_index(args[2])
    load = b.create(stencil.load, field=args[0], lb=lb, ub=ub,
                    res=tmp.unshaped_f64_temp(), loc=op.loc)
    b.replace(op, [load.res()])

def raiseStencilStore(op, b):
    if not isa(op.func().definingOp, tmp.stencil_store):
        return False
    args = op.args()
    lb = Attribute() if len(args) == 2 else get_index(args[2])
    ub = Attribute() if len(args) == 2 else get_index(args[3])
    b.create(stencil.store, temp=args[1], field=args[0], lb=lb, ub=ub,
             loc=op.loc)
    b.erase(op)

def raiseStencilApply(op, b):
    if not isa(op.func().definingOp, tmp.stencil_apply):
        return False
    args = op.args()
    # TODO more than one induction variable
    assert len(args) == 2 and isa(args[1].definingOp, tmp.stencil_apply_body)
    apply = b.create(stencil.apply, operands=[args[0]],
                     res=[tmp.unshaped_f64_temp()], loc=op.loc)
    bvm = BlockAndValueMapping()
    entry = apply.region().addEntryBlock([tmp.unshaped_f64_temp()])
    body = tmp.stencil_apply_body(args[1].definingOp).body().getBlock(0)
    for temp, obj in zip(entry.getArguments(), body.getArguments()):
        bvm[obj] = temp
    copy_into(entry, body, bvm)
    b.replace(op, apply.res())

def raiseStencilAccess(op, b):
    if not isa(op.idx().definingOp, tmp.stencil_index):
        return False
    access = b.create(stencil.access, temp=op.arg(), offset=get_index(op.idx()),
                      res=F64Type(), loc=op.loc)
    b.replace(op, [access.res()])

def raiseStdConstant(op, b):
    const = b.create(ConstantOp, value=F64Attr(op.value().getInt()),
                     loc=op.loc)
    b.replace(op, [const.result()])

def raiseStdUnary(op, b):
    if op.op().getValue() != "-":
        return False
    neg1 = b.create(ConstantOp, value=F64Attr(-1), loc=op.loc).result()
    mul = b.create(MulFOp, ty=F64Type(), lhs=neg1, rhs=op.arg(), loc=op.loc)
    b.replace(op, [mul.result()])

def raiseStdBinary(bin_op, std_op):
    def patternFcn(op, b):
        if op.op().getValue() != bin_op:
            return False
        stdOp = b.create(std_op, ty=F64Type(), lhs=op.lhs(), rhs=op.rhs(),
                         loc=op.loc)
        b.replace(op, [stdOp.result()])
    return Pattern(py.binary, patternFcn)

def raiseStencilReturn(op, b):
    if not isa(op.parentOp, stencil.apply):
        return False
    b.create(stencil.Return, operands=list(op.args()), loc=op.loc)
    b.erase(op)

def raiseStdReturn(op, b):
    if not isa(op.parentOp, FuncOp):
        return False
    b.create(ReturnOp, operands=list(op.args()), loc=op.loc)
    b.erase(op)

def raisePass(m):
    applyOptPatterns(m, [
        Pattern(py.load, raiseStencilModule),
        raiseStencilFcn("cast", tmp.stencil_assert),
        raiseStencilFcn("load", tmp.stencil_load),
        raiseStencilFcn("store", tmp.stencil_store),
        raiseStencilFcn("apply", tmp.stencil_apply),
        Pattern(py.make_tuple, raiseTupleOrListToIndex),
        Pattern(py.make_list, raiseTupleOrListToIndex),
        Pattern(py.load, raiseFunctionRef),
    ])
    target = ConversionTarget()
    target.addIllegalDialect(str(py.name))
    target.addIllegalDialect(str(tmp.name))
    applyOptPatterns(m, [
        Pattern(py.func, raiseStencilProgram),
        Pattern(py.call, raiseStencilAssert),
        Pattern(py.call, raiseStencilLoad),
        Pattern(py.call, raiseStencilApply),
        Pattern(py.call, raiseStencilStore),
        Pattern(py.constant, raiseStdConstant),
        Pattern(py.unary, raiseStdUnary),
        raiseStdBinary("+", AddFOp),
        raiseStdBinary("-", SubFOp),
        raiseStdBinary("*", MulFOp),
        Pattern(py.ret, raiseStencilReturn),
        Pattern(py.ret, raiseStdReturn),
        Pattern(py.subscript, raiseStencilAccess),
    ])
    applyPartialConversion(m, [], target)

################################################################################
# Public API
################################################################################

def program(func):
    node = ast.parse(inspect.getsource(func))
    m = StencilProgramVisitor().visit(node)
    varAllocPass(m)
    raisePass(m)
    verify(m)
    def wrapper():
        return m
    return wrapper
