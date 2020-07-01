#!/usr/local/bin/python3
import os
import sys

from antlr4 import *
from mlir import *

from LuaLexer import LuaLexer
from LuaListener import LuaListener
from LuaParser import LuaParser

import sys

def get_dialects(filename='lua.mlir'):
    m = parseSourceFile(filename)
    assert m, "failed to load dialects"
    dialects = registerDynamicDialects(m)
    return dialects[0], dialects[1], dialects[2]

lua, luac, luallvm = get_dialects()

class Generator:
    ############################################################################
    # Constructor and Helpers
    ############################################################################

    def __init__(self, filename:str, stream:CommonTokenStream):
        self.filename = filename
        self.stream = stream

    @staticmethod
    def addEntryBlock(region:Region):
        assert region != None, "no region currently set"
        assert len(region) == 0, "region already has a block"

        # All main blocks will have zero arguments
        return region.addEntryBlock([])

    def getStartLoc(self, ctx:ParserRuleContext):
        a, b = ctx.getSourceInterval()
        startTok = self.stream.get(a)
        return FileLineColLoc(self.filename, startTok.line, startTok.column)

    def getEndLoc(self, ctx:ParserRuleContext):
        a, b = ctx.getSourceInterval()
        endTok = self.stream.get(b)
        return FileLineColLoc(self.filename, endTok.line, endTok.column)

    ############################################################################
    # AST Walker
    ############################################################################

    def chunk(self, ctx:LuaParser.ChunkContext):
        # Create the Lua module
        module = ModuleOp()
        main = FuncOp("main", FunctionType(), self.getStartLoc(ctx))
        module.append(main)

        # Set the current region
        self.region = main.getBody()

        # Process the block
        self.block(ctx.block())

        # Return the module
        assert verify(module), "module failed to verify"
        return module, main

    def block(self, ctx:LuaParser.BlockContext):
        # Add an entry block to the enclosing region and set it
        self.block = self.addEntryBlock(self.region)
        self.builder = Builder()
        self.builder.insertAtStart(self.block)

        # Process the statements
        for stat in ctx.stat():
            self.stat(stat)

        # Process the return statement or insert an empty terminator
        if ctx.retstat():
            raise NotImplementedError("return statements not handled")
        else:
            self.builder.create(ReturnOp, loc=self.getEndLoc(ctx))

    def stat(self, ctx:LuaParser.StatContext):
        if ctx.assignlist():
            self.assignlist(ctx.assignlist())
        elif ctx.functioncall():
            self.functioncall(ctx.functioncall())
        elif ctx.label():
            raise NotImplementedError("label not implemented")
        elif ctx.breakstmt():
            raise NotImplementedError("breakstmt not implemented")
        elif ctx.gotostmt():
            raise NotImplementedError("gotostmt not implemented")
        elif ctx.enclosedblock():
            raise NotImplementedError("enclosedblock not implemented")
        elif ctx.whileloop():
            raise NotImplementedError("whileloop not implemented")
        elif ctx.repeatloop():
            raise NotImplementedError("repeatloop not implemented")
        elif ctx.conditionalchain():
            raise NotImplementedError("conditionalchain not implemented")
        elif ctx.numericfor():
            raise NotImplementedError("numericfor not implemented")
        elif ctx.genericfor():
            raise NotImplementedError("genericfor not implemented")
        elif ctx.namedfunctiondef():
            raise NotImplementedError("namedfunctiondef not implemented")
        elif ctx.localnamedfunctiondef():
            raise NotImplementedError("localnamedfunctiondef not implemented")
        elif ctx.localvarlist():
            raise NotImplementedError("localvarlist not implemented")
        else:
            raise ValueError("Unknown StatContext case")

    def assignlist(self, ctx:LuaParser.AssignlistContext):
        varList = self.varlist(ctx.varlist())
        expList = self.explist(ctx.explist())

        # Assign via concat and unpack
        concat = self.builder.create(lua.concat, vals=expList,
                                     loc=self.getStartLoc(ctx))
        unpack = self.builder.create(lua.unpack, pack=concat.pack(),
                                     vals=[lua.ref()] * len(varList),
                                     loc=self.getStartLoc(ctx))

        vals = unpack.vals()
        assert len(vals) == len(varList)
        for i in range(0, len(varList)):
            var = varList[i].definingOp.getAttr("var")
            self.builder.create(lua.assign, tgt=varList[i], val=vals[i],
                                var=var, loc=self.getStartLoc(ctx))

    def varlist(self, ctx:LuaParser.VarlistContext):
        varList = []
        for var in ctx.var():
            varList.append(self.var(var))
        return varList

    def explist(self, ctx:LuaParser.ExplistContext):
        expList = []
        for exp in ctx.exp():
            expList.append(self.exp(exp))
        return expList

    def nameAndArgs(self, ctx:LuaParser.NameAndArgsContext):
        assert ctx.NAME() == None, "colon operator unimplemented"
        return self.args(ctx.args())

    def args(self, ctx:LuaParser.ArgsContext):
        if ctx.explist():
            return self.explist(ctx.explist())
        elif ctx.tableconstructor():
            # Ensure return value is a list of values
            return [self.tableconstructor(ctx.tableconstructor())]
        elif ctx.string():
            return [self.string(ctx.string())]
        else:
            raise ValueError("Unknown ArgsContext case")

    def varOrExp(self, ctx:LuaParser.VarOrExpContext):
        if ctx.var():
            return self.var(ctx.var())
        elif ctx.exp():
            return self.exp(ctx.exp())
        else:
            raise ValueError("Unknown VarOrExpContext case")

    def var(self, ctx:LuaParser.VarContext):
        assert ctx.exp() == None and len(ctx.varSuffix()) == 0, \
            "only identifier variables are supported"

        alloc = self.builder.create(lua.get_or_alloc,
                                    var=StringAttr(ctx.NAME().getText()),
                                    loc=self.getStartLoc(ctx))
        return alloc.res()

    def exp(self, ctx:LuaParser.ExpContext):
        if ctx.nilvalue():
            return self.nilvalue(ctx.nilvalue())
        elif ctx.falsevalue():
            raise NotImplementedError("falsevalue not implemented")
        elif ctx.truevalue():
            raise NotImplementedError("truevalue not implemented")
        elif ctx.number():
            return self.number(ctx.number())
        elif ctx.string():
            raise NotImplementedError("string not implemented")
        elif ctx.elipsis():
            raise NotImplementedError("elipsis not implemented")
        elif ctx.functiondef():
            raise NotImplementedError("functiondef not implemented")
        elif ctx.prefixexp():
            return self.prefixexp(ctx.prefixexp())
        elif ctx.tableconstructor():
            raise NotImplementedError("tableconstructor not implemented")
        elif ctx.operatorUnary():
            raise NotImplementedError("operatorUnary not implemented")
        elif ctx.operatorPower():
            return self.operatorBinary(ctx, ctx.operatorPower())
        elif ctx.operatorMulDivMod():
            return self.operatorBinary(ctx, ctx.operatorMulDivMod())
        elif ctx.operatorAddSub():
            return self.operatorBinary(ctx, ctx.operatorAddSub())
        elif ctx.operatorStrcat():
            return self.operatorBinary(ctx, ctx.operatorStrcat())
        elif ctx.operatorComparison():
            return self.operatorBinary(ctx, ctx.operatorComparison())
        elif ctx.operatorAnd():
            return self.operatorBinary(ctx, ctx.operatorAnd())
        elif ctx.operatorOr():
            return self.operatorBinary(ctx, ctx.operatorOr())
        elif ctx.operatorBitwise():
            return self.operatorBinary(ctx, ctx.operatorBitwise())
        else:
            raise ValueError("Unknown ExpContext case")

    def nilvalue(self, ctx:LuaParser.NilvalueContext):
        nil = self.builder.create(lua.nil, loc=self.getStartLoc(ctx))
        return nil.res()

    def number(self, ctx:LuaParser.NumberContext):
        if ctx.INT():
            attr = I64Attr(int(ctx.INT().getText()))
        elif ctx.HEX():
            attr = I64Attr(int(ctx.HEX().getText(), 16))
        elif ctx.FLOAT():
            attr = F64Attr(float(ctx.FLOAT().getText()()))
        elif ctx.HEX_FLOAT():
            raise NotImplementedError("number HEX_FLOAT not implemented")
        else:
            raise NotImplementedError("Unknown NumberContext case")
        number = self.builder.create(lua.number, value=attr,
                                     loc=self.getStartLoc(ctx))
        return number.res()

    def prefixexp(self, ctx:LuaParser.PrefixexpContext):
        # A variable or expression, with optional trailing function calls
        # `myFcn (a, b, c) {a: b} "abc" {} "" (c, b, d)` is valid syntax
        val = self.varOrExp(ctx.varOrExp())
        allArgs = ctx.nameAndArgs()

        if len(allArgs) == 0:
            return val
        assert len(allArgs) == 1, "trailing function calls unimplemented"

        args = self.nameAndArgs(allArgs[0])
        concat = self.builder.create(lua.concat, vals=args,
                                     loc=self.getStartLoc(ctx))
        call = self.builder.create(lua.call, fcn=val, args=concat.pack(),
                                   loc=self.getStartLoc(ctx))
        # Return a value pack. This means that self.exp could return either a
        # value or value pack. Module verifier should catch accidental uses of
        # value pack as value.
        return call.rets()

    def functioncall(self, ctx:LuaParser.FunctioncallContext):
        # Identical to prefixexp, except len(allArgs) >= 1, and statement does
        # not return a value
        self.prefixexp(ctx)

    def operatorBinary(self, expCtx, opCtx):
        lhsVal = self.exp(expCtx.exp(0))
        rhsVal = self.exp(expCtx.exp(1))
        binary = self.builder.create(lua.binary, lhs=lhsVal, rhs=rhsVal,
                                     op=StringAttr(opCtx.getText()),
                                     loc=self.getStartLoc(opCtx))
        return binary.res()


class AllocVisitor:
    def __init__(self):
        self.scope = {}
        self.builder = Builder()
        self.removed = set()

    def visitAll(self, ops:Region):
        worklist = []
        for op in ops: # iterator is invalidated if an op is removed
            worklist.append(op)
        for op in worklist:
            if op not in self.removed:
                self.visit(op)

    def visit(self, op:Operation):
        if isa(op, lua.get_or_alloc):
            self.visitGetOrAlloc(lua.get_or_alloc(op))
        elif isa(op, lua.assign):
            self.visitAssign(lua.assign(op))
        elif isa(op, lua.call):
            self.visitCall(lua.call(op))

    def visitGetOrAlloc(self, op:lua.get_or_alloc):
        if op.var() not in self.scope:
            self.builder.insertBefore(op)
            alloc = self.builder.create(lua.alloc, var=op.var(), loc=op.loc)
            self.scope[op.var()] = alloc.res()
        self.builder.replace(op, [self.scope[op.var()]])
        self.removed.add(op)

    def visitAssign(self, op:lua.assign):
        assert op.var() in self.scope
        self.scope[op.var()] = op.res()

    def visitCall(self, op:lua.call):
        if op.rets().useEmpty():
            self.builder.insertAfter(op)
            # unused return values
            self.builder.create(luac.delete_pack, pack=op.rets(), loc=op.loc)


def elideConcatAndUnpack(op:lua.unpack, rewriter:Builder):
    parent = op.pack().definingOp
    if not isa(parent, lua.concat):
        return False
    concat = lua.concat(parent)
    trivialUnpack = all(val.type == lua.ref() for val in concat.vals())
    if not trivialUnpack:
        return False
    newVals = []
    for i in range(0, min(len(concat.vals()), len(op.vals()))):
        newVals.append(concat.vals()[i])
    nil = rewriter.create(lua.nil, loc=op.loc)
    for i in range(len(newVals), len(op.vals())):
        newVals.append(nil.res())
    rewriter.replace(op, newVals)
    return True

lua_builtins = set(["print"])
def raiseBuiltins(op:lua.alloc, rewriter:Builder):
    if op.var().getValue() not in lua_builtins:
        return False
    builtin = rewriter.create(lua.builtin, var=op.var(), loc=op.loc)
    rewriter.replace(op, [builtin.val()])
    return True

def elideAssign(op:lua.assign, rewriter:Builder):
    # Since scopes not implemented, all assignments are trivial
    rewriter.replace(op, [op.val()])
    return True

def varAllocPass(main:FuncOp):
    # Main function has one Sized<1> region
    region = main.getBody()
    ops = region.getBlock(0)

    # Non-trivial passes
    AllocVisitor().visitAll(ops)

    # Trivial passes as patterns
    applyOptPatterns(main, [
        Pattern(lua.unpack, elideConcatAndUnpack, [lua.nil]),
        Pattern(lua.alloc, raiseBuiltins, [lua.builtin]),
        Pattern(lua.assign, elideAssign),
    ])

def getWrapperFor(numberTy, opCls):
    def wrapFcn(op:lua.number, rewriter:Builder):
        if op.value().type != numberTy:
            return False
        const = rewriter.create(ConstantOp, value=op.value(), loc=op.loc)
        wrap = rewriter.create(opCls, num=const.result(), loc=op.loc)
        rewriter.replace(op, [wrap.res()])
        return True

    return wrapFcn

def getExpanderFor(opStr, binOpCls):
    def expandFcn(op:lua.binary, rewriter:Builder):
        if op.op().getValue() != opStr:
            return False
        binOp = rewriter.create(binOpCls, lhs=op.lhs(), rhs=op.rhs(),
                                loc=op.loc)
        rewriter.replace(op, [binOp.res()])
        return True

    return expandFcn

def allocAndSet(setOp):
    def matchFcn(op, rewriter:Builder):
        alloc = rewriter.create(luac.alloc, loc=op.loc)
        const = rewriter.create(ConstantOp, value=luac.type_num(), loc=op.loc)
        rewriter.create(setOp, tgt=alloc.res(), num=op.num(), loc=op.loc)
        rewriter.create(luac.set_type, tgt=alloc.res(), ty=const.result(),
                        loc=op.loc)
        rewriter.replace(op, [alloc.res()])
        return True

    return matchFcn

def setToNil(op:lua.nil, rewriter:Builder):
    alloc = rewriter.create(luac.alloc, loc=op.loc)
    const = rewriter.create(ConstantOp, value=luac.type_nil(), loc=op.loc)
    rewriter.create(luac.set_type, tgt=alloc.res(), ty=const.result(),
                    loc=op.loc)
    rewriter.replace(op, [alloc.res()])
    return True

def expandConcat(op:lua.concat, rewriter:Builder):
    assert op.pack().hasOneUse(), "value pack can only be used once"
    fixedSz = sum(1 if val.type == lua.ref() else 0 for val in op.vals())
    const = rewriter.create(ConstantOp, value=I64Attr(fixedSz), loc=op.loc)
    szVar = const.result()
    for val in op.vals():
        if val.type == lua.pack():
            getSz = rewriter.create(luac.pack_get_size, pack=val, loc=op.loc)
            addI = rewriter.create(AddIOp, lhs=szVar, rhs=getSz.sz(),
                                   ty=IntegerType(64), loc=op.loc)
            szVar = addI.result()

    newPack = rewriter.create(luac.new_pack, rsv=szVar, loc=op.loc)
    for val in op.vals():
        if val.type == lua.ref():
            rewriter.create(luac.pack_push, pack=newPack.pack(), val=val,
                            loc=op.loc)
        else:
            rewriter.create(luac.pack_push_all, pack=newPack.pack(), vals=val,
                            loc=op.loc)
            rewriter.create(luac.delete_pack, pack=val, loc=op.loc)
    rewriter.replace(op, [newPack.pack()])
    return True

def expandUnpack(op:lua.unpack, rewriter:Builder):
    assert op.pack().hasOneUse(), "value pack can only be used once"
    newVals = []
    for val in op.vals():
        pull = rewriter.create(luac.pack_pull_one, pack=op.pack(), loc=op.loc)
        newVals.append(pull.val())
    rewriter.replace(op, newVals)
    rewriter.create(luac.delete_pack, pack=op.pack(), loc=op.loc)
    return True

def expandCall(op:lua.call, rewriter:Builder):
    getAddr = rewriter.create(luac.get_fcn_addr, fcn=op.fcn(), loc=op.loc)
    icall = rewriter.create(CallIndirectOp, callee=getAddr.fcn_addr(),
                            operands=[op.args()], loc=op.loc)
    rewriter.replace(op, icall.results())
    return True

def lowerAlloc(op:lua.alloc, rewriter:Builder):
    nil = rewriter.create(lua.nil, loc=op.loc)
    rewriter.replace(op, [nil.res()])
    return True


class ConstantElVisitor:
    def __init__(self):
        self.consts = {}
        self.removed = set()
        self.builder = Builder()

    def visitAll(self, ops:Region):
        worklist = []
        for op in ops: # iterator is invalidated if an op is removed
            worklist.append(op)
        for op in worklist:
            if op not in self.removed:
                self.visit(op)

    def visit(self, op:Operation):
        if isa(op, ConstantOp):
            self.visitConstantOp(ConstantOp(op))

    def visitConstantOp(self, op:ConstantOp):
        if op.value() in self.consts:
            self.builder.insertBefore(op)
            self.builder.replace(op, [self.consts[op.value()]])
        else:
            self.consts[op.value()] = op.result()

def convertToLibCall(module:ModuleOp, funcName:str):
    def convertFcn(op:Operation, rewriter:Builder):
        rawOp = module.lookup(funcName)
        assert rawOp, "cannot find lib.mlir function '" + funcName + "'"
        call = rewriter.create(CallOp, callee=rawOp,
                               operands=op.getOperands(), loc=op.loc)
        results = call.getResults()
        if len(results) == 0:
            rewriter.erase(op)
        else:
            rewriter.replace(op, results)
        return True

    return convertFcn

def lowerToLuac(module:ModuleOp):
    target = ConversionTarget()

    target.addLegalDialect(luac)
    target.addLegalDialect("std")
    target.addLegalOp(FuncOp)

    target.addIllegalOp(luac.wrap_int)
    target.addIllegalOp(luac.wrap_real)

    target.addLegalOp(lua.builtin)
    target.addLegalOp(ModuleOp)

    patterns = [
        Pattern(lua.alloc, lowerAlloc, [lua.nil]),

        Pattern(lua.number, getWrapperFor(luac.integer(), luac.wrap_int),
                [ConstantOp, luac.wrap_int]),
        Pattern(lua.number, getWrapperFor(luac.real(), luac.wrap_real),
                [ConstantOp, luac.wrap_real]),

        Pattern(lua.binary, getExpanderFor("+", luac.add), [luac.add]),
        Pattern(lua.binary, getExpanderFor("-", luac.sub), [luac.sub]),
        Pattern(lua.binary, getExpanderFor("*", luac.mul), [luac.mul]),

        Pattern(luac.wrap_int, allocAndSet(luac.set_int64_val),
                [luac.alloc, luac.set_int64_val, luac.set_type, ConstantOp]),
        Pattern(luac.wrap_real, allocAndSet(luac.set_double_val),
                [luac.alloc, luac.set_double_val, luac.set_type, ConstantOp]),
        Pattern(lua.nil, setToNil, [luac.alloc, luac.set_type]),

        Pattern(lua.concat, expandConcat, [luac.new_pack, luac.pack_push,
                                           luac.pack_push_all,
                                           luac.pack_get_size]),
        Pattern(lua.unpack, expandUnpack, [luac.pack_pull_one,
                                           luac.delete_pack]),
        Pattern(lua.call, expandCall, [luac.get_fcn_addr, CallIndirectOp]),
    ]

    for func in module.getOps(FuncOp):
        applyFullConversion(func, patterns, target)


def valueConverter(ty:Type):
    if ty != lua.ref():
        return None
    return luallvm.ref()

def packConverter(ty:Type):
    if ty != lua.pack():
        return None
    return luallvm.pack()

def convertToFunc(module:ModuleOp, funcName:str):
    def convertFcn(op:Operation, rewriter:Builder):
        rawOp = module.lookup(funcName)
        assert rawOp, "cannot find libc function '" + funcName + "'"
        llvmCall = rewriter.create(LLVMCallOp, func=LLVMFuncOp(rawOp),
                                   operands=op.getOperands(), loc=op.loc)
        results = llvmCall.getResults()
        if len(results) == 0:
            rewriter.erase(op)
        else:
            rewriter.replace(op, results)
        return True

    return convertFcn

def luaToLLVM(module):
    def convert(opCls, funcName:str):
        return Pattern(opCls, convertToLibCall(module, funcName), [CallOp])

    def convertLibc(opCls, funcName:str):
        return Pattern(opCls, convertToFunc(module, funcName), [LLVMCallOp])

    def builtin(varName:str):
        return Pattern(lua.builtin,
                       convertToFunc(module, "lua_builtin_" + varName),
                       [LLVMCallOp])


    llvmPats = [
        convert(luac.add, "lua_add"),
        convert(luac.sub, "lua_sub"),
        convert(luac.mul, "lua_mul"),

        convertLibc(luac.alloc, "lua_alloc"),
        convertLibc(luac.get_type, "lua_get_type"),
        convertLibc(luac.set_type, "lua_set_type"),
        convertLibc(luac.get_int64_val, "lua_get_int64_val"),
        convertLibc(luac.set_int64_val, "lua_set_int64_val"),
        convertLibc(luac.get_double_val, "lua_get_double_val"),
        convertLibc(luac.set_double_val, "lua_set_double_val"),
        convertLibc(luac.get_fcn_addr, "lua_get_fcn_addr"),
        convertLibc(luac.get_value_union, "lua_get_value_union"),
        convertLibc(luac.set_value_union, "lua_set_value_union"),
        convertLibc(luac.is_int, "lua_is_int"),
        convertLibc(luac.new_pack, "lua_new_pack"),
        convertLibc(luac.delete_pack, "lua_delete_pack"),
        convertLibc(luac.pack_push, "lua_pack_push"),
        convertLibc(luac.pack_pull_one, "lua_pack_pull_one"),
        convertLibc(luac.pack_push_all, "lua_pack_push_all"),
        convertLibc(luac.pack_get_size, "lua_pack_get_size"),

        builtin("print"),
    ]
    lowerToLLVM(module, LLVMConversionTarget(), llvmPats,
                [valueConverter, packConverter])


def main():
    if len(sys.argv) != 2:
        print("Usage: luac.py <lua_file>")
        return
    filename = sys.argv[1]
    with open(filename, 'r') as file:
        contents = file.read()

    lexer = LuaLexer(InputStream(contents))
    stream = CommonTokenStream(lexer)
    parser = LuaParser(stream)

    generator = Generator(filename, stream)

    module, main = generator.chunk(parser.chunk())
    varAllocPass(main)

    lib = parseSourceFile("lib.mlir")
    for func in lib.getOps(FuncOp):
        module.append(func.clone())

    lowerSCFToStandard(module)
    lowerToLuac(module)
    ConstantElVisitor().visitAll(main.getBody().getBlock(0))

    os.system("clang -S -emit-llvm lib.c -o lib.ll")
    os.system("mlir-translate -import-llvm lib.ll -o libc.mlir")
    libc = parseSourceFile("libc.mlir")
    for glob in libc.getOps(LLVMGlobalOp):
        module.append(glob.clone())
    for func in libc.getOps(LLVMFuncOp):
        module.append(func.clone())

    luaToLLVM(module)
    verify(module)

    with open("main.mlir", "w") as f:
        stdout = sys.stdout
        sys.stdout = f
        print(module)
        sys.stdout = stdout

    os.system("clang++ -c builtins.cpp -o builtins.o -g")
    os.system("mlir-translate -mlir-to-llvmir main.mlir -o main.ll")
    os.system("clang -c main.ll -o main.o")
    os.system("ld main.o builtins.o -lc -lc++ -o main")

if __name__ == '__main__':
    main()
