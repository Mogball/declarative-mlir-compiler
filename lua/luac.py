#!/usr/bin/python3
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

################################################################################
# Front-End: Parser and AST Walker / MLIR Generator
################################################################################

class Generator:
    #### Helpers
    ###############

    def __init__(self, filename:str, stream:CommonTokenStream):
        self.filename = filename
        self.stream = stream

    def getStartLoc(self, ctx:ParserRuleContext):
        a, b = ctx.getSourceInterval()
        startTok = self.stream.get(a)
        return FileLineColLoc(self.filename, startTok.line, startTok.column)

    def getEndLoc(self, ctx:ParserRuleContext):
        a, b = ctx.getSourceInterval()
        endTok = self.stream.get(b)
        return FileLineColLoc(self.filename, endTok.line, endTok.column)

    def handleAssignList(self, varList, expList, loc):
        concat = self.builder.create(lua.concat, vals=expList, loc=loc)
        unpack = self.builder.create(lua.unpack, pack=concat.pack(), loc=loc,
                                     vals=[lua.ref()] * len(varList))
        vals = unpack.vals()
        assert len(vals) == len(varList)
        for i in range(0, len(varList)):
            var = varList[i].definingOp.getAttr("var")
            self.builder.create(lua.assign, tgt=varList[i], val=vals[i],
                                var=var, loc=loc)

    def handleBinaryOp(self, expCtx, opCtx):
        lhsVal = self.exp(expCtx.exp(0))
        rhsVal = self.exp(expCtx.exp(1))
        binary = self.builder.create(lua.binary, lhs=lhsVal, rhs=rhsVal,
                                     op=StringAttr(opCtx.getText()),
                                     loc=self.getStartLoc(opCtx))
        return binary.res()

    #### AST Walker
    ###############

    def chunk(self, ctx:LuaParser.ChunkContext):
        # Create the Lua module
        module = ModuleOp()
        main = FuncOp("main", FunctionType(), self.getStartLoc(ctx))
        module.append(main)

        # Set the current region
        self.region = main.getBody()
        self.curBlock = self.region.addEntryBlock([])
        self.builder = Builder()

        # Process the block
        self.builder.insertAtStart(self.curBlock)
        self.block(ctx.block())
        self.builder.create(ReturnOp, loc=self.getEndLoc(ctx))

        # Return the module
        assert verify(module), "module failed to verify"
        return module, main

    def block(self, ctx:LuaParser.BlockContext):
        # Process the statements
        for stat in ctx.stat():
            self.stat(stat)

        # Process the return statement
        assert ctx.retstat() == None, "return statements not implemented"

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
            self.numericfor(ctx.numericfor())
        elif ctx.genericfor():
            raise NotImplementedError("genericfor not implemented")
        elif ctx.namedfunctiondef():
            self.namedfunctiondef(ctx.namedfunctiondef())
        elif ctx.localnamedfunctiondef():
            raise NotImplementedError("localnamedfunctiondef not implemented")
        elif ctx.localvarlist():
            self.localvarlist(ctx.localvarlist())
        else:
            raise ValueError("Unknown StatContext case")

    def assignlist(self, ctx:LuaParser.AssignlistContext):
        self.handleAssignList(self.varlist(ctx.varlist()),
                              self.explist(ctx.explist()),
                              self.getStartLoc(ctx))

    def localvarlist(self, ctx:LuaParser.LocalvarlistContext):
        varList = []
        loc = self.getStartLoc(ctx)
        for varName in ctx.namelist().NAME():
            lalloc = self.builder.create(lua.alloc_local, loc=loc,
                                         var=StringAttr(varName.getText()))
            varList.append(lalloc.res())

        if ctx.explist():
            self.handleAssignList(varList, self.explist(ctx.explist()), loc)

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
            return []

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
            return self.handleBinaryOp(ctx, ctx.operatorPower())
        elif ctx.operatorMulDivMod():
            return self.handleBinaryOp(ctx, ctx.operatorMulDivMod())
        elif ctx.operatorAddSub():
            return self.handleBinaryOp(ctx, ctx.operatorAddSub())
        elif ctx.operatorStrcat():
            return self.handleBinaryOp(ctx, ctx.operatorStrcat())
        elif ctx.operatorComparison():
            return self.handleBinaryOp(ctx, ctx.operatorComparison())
        elif ctx.operatorAnd():
            return self.handleBinaryOp(ctx, ctx.operatorAnd())
        elif ctx.operatorOr():
            return self.handleBinaryOp(ctx, ctx.operatorOr())
        elif ctx.operatorBitwise():
            return self.handleBinaryOp(ctx, ctx.operatorBitwise())
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

    def numericfor(self, ctx:LuaParser.NumericforContext):
        loc = self.getStartLoc(ctx)

        # Process expressions for the loop limits
        lower = self.exp(ctx.exp(0))
        upper = self.exp(ctx.exp(1))
        if len(ctx.exp()) == 2:
            num = self.builder.create(lua.number, value=I64Attr(1), loc=loc)
            step = num.res()
        else:
            step = self.exp(ctx.exp(2))

        # Create the loop
        varName = StringAttr(ctx.NAME().getText())
        loop = self.builder.create(lua.numeric_for, lower=lower, upper=upper,
                                   step=step, loc=loc, ivar=varName)

        # Save previous block and add entry block
        prevBlock = self.curBlock
        self.curBlock = loop.region().addEntryBlock([lua.ref()])
        self.builder.insertAtStart(self.curBlock)
        # Process the block and restore builder
        self.block(ctx.block())
        self.builder.create(lua.end, loc=loc)
        self.curBlock = prevBlock
        self.builder.insertAtEnd(self.curBlock)

    def funcname(self, ctx:LuaParser.FuncnameContext):
        assert len(ctx.NAME()) == 1, "only simple function names supported"
        return StringAttr(ctx.NAME(0).getText())

    def parlist(self, ctx:LuaParser.ParlistContext):
        assert ctx.elipsis() == None, "variadic functions unsupported"
        return ArrayAttr([StringAttr(n.getText()) for n in ctx.namelist().NAME()])

    def funcbody(self, ctx:LuaParser.FuncbodyContext):
        return self.parlist(ctx.parlist()) if ctx.parlist() else ArrayAttr([])

    def namedfunctiondef(self, ctx:LuaParser.NamedfunctiondefContext):
        name = self.funcname(ctx.funcname())
        params = self.funcbody(ctx.funcbody())
        loc = self.getStartLoc(ctx)
        fcnDef = self.builder.create(lua.function_def, params=params, loc=loc)
        alloc = self.builder.create(lua.get_or_alloc, var=name, loc=loc)
        self.builder.create(lua.assign, tgt=alloc.res(), val=fcnDef.fcn(),
                            var=name, loc=loc)

        prevBlock = self.curBlock
        self.curBlock = fcnDef.region().addEntryBlock([lua.ref()] * len(params))
        self.builder.insertAtStart(self.curBlock)

        self.block(ctx.funcbody().block())
        self.builder.create(lua.ret, vals=[], loc=loc)
        self.curBlock = prevBlock
        self.builder.insertAtEnd(self.curBlock)

################################################################################
# Front-End: Variable Allocation and SSA
################################################################################

class ScopedMap:
    def __init__(self):
        self.scopes = [{}]

    def contains(self, e):
        for scope in reversed(self.scopes):
            if e in scope:
                return True
        return False

    def lookup(self, k):
        for scope in reversed(self.scopes):
            if k in scope:
                return scope[k]
        return None

    def set_local(self, k, e):
        scope = self.scopes[len(self.scopes) - 1]
        scope[k] = e

    def set_global(self, k, e):
        scope = self.scopes[0]
        scope[k] = e

    def push_scope(self):
        self.scopes.append({})

    def pop_scope(self):
        self.scopes.pop()

def walkInOrder(op, func):
    func(op)
    for region in op.getRegions():
        for block in region:
            for o in block:
                walkInOrder(o, func)


class AllocVisitor:
    def __init__(self):
        self.scope = ScopedMap()
        self.builder = Builder()
        self.removed = set()

    def visitAll(self, main:FuncOp):
        worklist = []
        self.global_block = main.getBody().getBlock(0)
        walkInOrder(main, lambda op : worklist.append(op))
        for op in worklist:
            if op not in self.removed:
                self.visit(op)

    def visit(self, op:Operation):
        if isa(op, lua.get_or_alloc):
            self.visitGetOrAlloc(lua.get_or_alloc(op))
        elif isa(op, lua.alloc_local):
            self.visitAllocLocal(lua.alloc_local(op))
        elif isa(op, lua.numeric_for):
            self.visitNumericFor(lua.numeric_for(op))
        elif isa(op, lua.assign):
            self.visitAssign(lua.assign(op))
        elif isa(op, lua.call):
            self.visitCall(lua.call(op))
        # TODO var alloc pass function_def

        if op.getNumRegions() != 0:
            self.scope.push_scope()
        elif isa(op, lua.end):
            self.scope.pop_scope()

    def visitGetOrAlloc(self, op:lua.get_or_alloc):
        if not self.scope.contains(op.var()):
            self.builder.insertAtStart(self.global_block)
            alloc = self.builder.create(lua.alloc, var=op.var(), loc=op.loc)
            self.scope.set_global(op.var(), alloc.res())
        self.builder.replace(op, [self.scope.lookup(op.var())])
        self.removed.add(op)

    def visitAllocLocal(self, op:lua.alloc_local):
        # overwrite previous value
        self.builder.insertBefore(op)
        alloc = self.builder.create(lua.alloc, var=op.var(), loc=op.loc)
        self.scope.set_local(op.var(), alloc.res())
        self.builder.replace(op, [alloc.res()])
        self.removed.add(op)

    def visitNumericFor(self, op:lua.numeric_for):
        self.scope.set_local(op.ivar(), op.region().getBlock(0).getArgument(0))

    def visitAssign(self, op:lua.assign):
        assert self.scope.contains(op.var())
        # TODO set this at the highest dominating scope
        self.scope.set_local(op.var(), op.res())

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

def scopeUseDominates(tgtOp, valOp):
    if valOp == None:
        return False
    while tgtOp and not isa(tgtOp, FuncOp):
        if tgtOp.parentOp == valOp.parentOp:
            return True
        tgtOp = tgtOp.parentOp
    return False

def elideAssign(op:lua.assign, rewriter:Builder):
    if scopeUseDominates(op.tgt().definingOp, op.val().definingOp):
        rewriter.replace(op, [op.val()])
        return True
    return False

def varAllocPass(main:FuncOp):
    # Non-trivial passes
    AllocVisitor().visitAll(main)

    # Trivial passes as patterns
    applyOptPatterns(main, [
        Pattern(lua.unpack, elideConcatAndUnpack, [lua.nil]),
        Pattern(lua.alloc, raiseBuiltins, [lua.builtin]),
        Pattern(lua.assign, elideAssign),
    ])

################################################################################
# IR: Dialect Conversion to SCF
################################################################################

def copyInto(newBlk, oldBlk, termCls, bvm=BlockAndValueMapping()):
    for oldOp in oldBlk:
        if not isa(oldOp, termCls):
            newBlk.append(oldOp.clone(bvm))

def lowerNumericFor(op:lua.numeric_for, rewriter:Builder):
    def toIndex(val):
        iv = rewriter.create(luac.get_int64_val, tgt=val, loc=op.loc)
        i = rewriter.create(IndexCastOp, source=iv.num(), type=IndexType(),
                            loc=op.loc)
        return i.result()
    # this is a little messy
    upper = toIndex(op.upper())
    const = rewriter.create(ConstantOp, value=IndexAttr(1), loc=op.loc)
    incr = rewriter.create(AddIOp, lhs=upper, rhs=const.result(),
                           ty=IndexType(), loc=op.loc)
    loop = rewriter.create(ForOp, lowerBound=toIndex(op.lower()),
                           upperBound=incr.result(), step=toIndex(op.step()),
                           loc=op.loc)
    loop.region().getBlock(0).erase()
    loopBlk = loop.region().addEntryBlock([IndexType()])
    forBlk = op.region().getBlock(0)

    rewriter.insertAtStart(loopBlk)
    i = rewriter.create(IndexCastOp, source=loop.getInductionVar(),
                        type=luac.integer(), loc=op.loc)
    val = rewriter.create(luac.wrap_int, num=i.result(), loc=op.loc)
    bvm = BlockAndValueMapping()
    bvm[forBlk.getArgument(0)] = val.res()
    copyInto(loopBlk, forBlk, lua.end, bvm)
    rewriter.insertAtEnd(loopBlk)
    rewriter.create(YieldOp, loc=op.loc)
    rewriter.erase(op)
    return True

anon_name_counter = 0
def lowerFunctionDef(module:ModuleOp):
    def lowerFcn(op:lua.function_def, rewriter:Builder):
        captures = set()
        def collectCaptures(o):
            for val in o.getOperands():
                if not op.isProperAncestor(val.definingOp):
                    captures.add(val)
        walkInOrder(op, collectCaptures)
        captures = list(captures)
        concatCaps = rewriter.create(lua.concat, vals=captures, loc=op.loc)

        global anon_name_counter
        name = "lua_anon_fcn_" + str(anon_name_counter)
        anon_name_counter += 1
        func = FuncOp(name, luac.pack_fcn(), op.loc)
        module.append(func)
        block = func.getBody().addEntryBlock([lua.pack(), lua.pack()])
        rewriter.insertAtStart(block)
        unpack = rewriter.create(lua.unpack, pack=block.getArgument(0),
                                 vals=[lua.ref()] * len(captures), loc=op.loc)

        bvm = BlockAndValueMapping()
        for i in range(0, len(captures)):
            bvm[captures[i]] = unpack.vals()[i]
        fcnBlk = op.region().getBlock(0)
        copyInto(block, fcnBlk, lua.ret, bvm)

        ret = lua.ret(fcnBlk.getTerminator())
        rewriter.insertAtEnd(block)
        concat = rewriter.create(lua.concat, vals=ret.getOperands(), loc=op.loc)
        rewriter.create(ReturnOp, operands=[concat.pack()], loc=op.loc)
        rewriter.insertBefore(op)

        fcnAddr = rewriter.create(ConstantOp, value=FlatSymbolRefAttr(name),
                                  ty=luac.pack_fcn(), loc=op.loc)
        alloc = rewriter.create(luac.alloc, loc=op.loc)
        const = rewriter.create(ConstantOp, value=luac.type_fcn(), loc=op.loc)
        rewriter.create(luac.set_type, tgt=alloc.res(), ty=const.result(),
                        loc=op.loc)
        rewriter.create(luac.set_fcn_addr, fcn=alloc.res(),
                        fcn_addr=fcnAddr.result(), loc=op.loc)
        rewriter.create(luac.set_capture_pack, fcn=alloc.res(),
                        pack=concatCaps.pack(), loc=op.loc)
        rewriter.replace(op, [alloc.res()])
        return True

    return lowerFcn

def cfExpand(module:ModuleOp, main:FuncOp):
    applyOptPatterns(main, [
        Pattern(lua.numeric_for, lowerNumericFor),
        Pattern(lua.function_def, lowerFunctionDef(module)),
    ])

################################################################################
# IR: Dialect Conversion to StandardOps
################################################################################

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
    if len(op.vals()) == 0:
        return False
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

def expandEmptyConcat(op:lua.concat, rewriter:Builder):
    if len(op.vals()) != 0:
        return False
    const = rewriter.create(ConstantOp, value=I64Attr(1), loc=op.loc)
    newPack = rewriter.create(luac.new_pack, rsv=const.result(), loc=op.loc)
    nil = rewriter.create(lua.nil, loc=op.loc)
    rewriter.create(luac.pack_push, pack=newPack.pack(), val=nil.res(), op=op.loc)
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
    getPack = rewriter.create(luac.get_capture_pack, fcn=op.fcn(), loc=op.loc)
    icall = rewriter.create(CallIndirectOp, callee=getAddr.fcn_addr(),
                            operands=[getPack.pack(), op.args()], loc=op.loc)
    rewriter.replace(op, icall.results())
    return True

def lowerAlloc(op:lua.alloc, rewriter:Builder):
    nil = rewriter.create(lua.nil, loc=op.loc)
    rewriter.replace(op, [nil.res()])
    return True

def expandAssign(op:lua.assign, rewriter:Builder):
    getType = rewriter.create(luac.get_type, tgt=op.val(), loc=op.loc)
    getU = rewriter.create(luac.get_value_union, tgt=op.val(), loc=op.loc)
    rewriter.create(luac.set_type, tgt=op.tgt(), ty=getType.ty(), loc=op.loc)
    rewriter.create(luac.set_value_union, tgt=op.tgt(), u=getU.u(), loc=op.loc)
    rewriter.replace(op, [op.tgt()])
    return True

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
    target.addLegalDialect("loop")
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
                                           luac.pack_push_all, luac.pack_get_size]),
        Pattern(lua.concat, expandEmptyConcat, [luac.new_pack, luac.pack_push,
                                                lua.nil]),
        Pattern(lua.unpack, expandUnpack, [luac.pack_pull_one,
                                           luac.delete_pack]),
        Pattern(lua.call, expandCall, [luac.get_fcn_addr, CallIndirectOp,
                                       luac.get_capture_pack]),
        Pattern(lua.assign, expandAssign, [luac.set_type, luac.set_value_union,
                                           luac.get_type, luac.get_value_union]),
    ]

    for func in module.getOps(FuncOp):
        applyFullConversion(func, patterns, target)

################################################################################
# IR: Dialect Conversion to LLVM IR
################################################################################

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
        convertLibc(luac.set_fcn_addr, "lua_set_fcn_addr"),
        convertLibc(luac.get_capture_pack, "lua_get_capture_pack"),
        convertLibc(luac.set_capture_pack, "lua_set_capture_pack"),
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
    cfExpand(module, main)

    #lib = parseSourceFile("lib.mlir")
    #for func in lib.getOps(FuncOp):
    #    module.append(func.clone())

    #lowerSCFToStandard(module)
    #lowerToLuac(module)

    #os.system("clang -S -emit-llvm lib.c -o lib.ll -O2")
    #os.system("mlir-translate -import-llvm lib.ll -o libc.mlir")
    #libc = parseSourceFile("libc.mlir")
    #for glob in libc.getOps(LLVMGlobalOp):
    #    module.append(glob.clone())
    #for func in libc.getOps(LLVMFuncOp):
    #    module.append(func.clone())

    #runAllOpts(module)

    #luaToLLVM(module)
    #verify(module)

    #with open("main.mlir", "w") as f:
    #    stdout = sys.stdout
    #    sys.stdout = f
    #    print(module)
    #    sys.stdout = stdout

    #os.system("clang++ -c builtins.cpp -o builtins.o -g -O2")
    #os.system("mlir-translate -mlir-to-llvmir main.mlir -o main.ll")
    #os.system("clang -c main.ll -o main.o -g -O2")
    #os.system("clang main.o builtins.o -o main -g -O2")

    print(module)
    verify(module)

if __name__ == '__main__':
    main()
