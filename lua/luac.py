#!/usr/bin/python3
import os
import sys

from antlr4 import *
from mlir import *

from LuaLexer import LuaLexer
from LuaListener import LuaListener
from LuaParser import LuaParser

import sys

################################################################################
# Initialization
################################################################################

def get_dialects(filename='lua.mlir'):
    m = parseSourceFile(filename)
    assert m, "failed to load dialects"
    dialects = registerDynamicDialects(m)
    return dialects[0], dialects[1], dialects[2], dialects[3]

lua, luaopt, luac, luallvm = get_dialects()

################################################################################
# Generic Helpers
################################################################################

def neverWrittenTo(val):
    return all(val not in getWriteEffectingValues(use)
               for use in val.getOpUses())

def licmDefinedOutside(op, val):
    return True

def licmCanHoist(op):
    return all(neverWrittenTo(result) for result in op.getResults())

def alwaysTrue(op):
    return True

################################################################################
# Front-End: Parser and AST Walker / MLIR Generator
################################################################################

def makeConcat(b, vals, tail, loc, concatLike=lua.concat):
    concat = b.create(concatLike, vals=vals, tail=tail, loc=loc)
    concat.setAttr("operand_segment_sizes", DenseIntElementsAttr(
        VectorType([2], I64Type(), loc), [len(vals), len(tail)]))
    return concat

class Generator:
    #########################################################
    # Helpers                                               #
    #########################################################

    def __init__(self, filename:str, stream:CommonTokenStream):
        self.filename = filename
        self.stream = stream
        self.builder = Builder()
        self.blockStack = []
        self.curBlock = None

    def getStartLoc(self, ctx:ParserRuleContext):
        a, b = ctx.getSourceInterval()
        startTok = self.stream.get(a)
        return FileLineColLoc(self.filename, startTok.line, startTok.column)

    def getEndLoc(self, ctx:ParserRuleContext):
        a, b = ctx.getSourceInterval()
        endTok = self.stream.get(b)
        return FileLineColLoc(self.filename, endTok.line, endTok.column)

    def handleBool(self, loc, val):
        b = self.builder.create(lua.boolean, value=I1Attr(val), loc=loc)
        return b.res()

    def handleBinaryOp(self, expCtx, opCtx):
        loc = self.getStartLoc(opCtx)
        lhsVal = self.exp(expCtx.exp(0))
        rhsVal = self.exp(expCtx.exp(1))
        binary = self.builder.create(lua.binary, lhs=lhsVal, rhs=rhsVal,
                                     op=StringAttr(opCtx.getText()), loc=loc)
        return binary.res()

    def handleFunctionLike(self, val, ctx, unpackLast=False):
        allArgs = ctx.nameAndArgs()

        if len(allArgs) == 0:
            return val
        loc = self.getStartLoc(ctx)
        for i in range(0, len(allArgs)):
            args = self.nameAndArgs(allArgs[i])
            call = self.builder.create(lua.call, fcn=val, args=args, loc=loc)
            if not unpackLast and i == len(allArgs) - 1:
                return call.rets()
            unpack = self.builder.create(lua.unpack, pack=call.rets(),
                                         vals=[lua.val()], loc=loc)
            val = unpack.vals()[0]
        assert unpackLast
        return val

    def pushBlock(self, block):
        self.blockStack.append(self.curBlock)
        self.curBlock = block
        self.builder.insertAtStart(self.curBlock)

    def popBlock(self):
        self.curBlock = self.blockStack.pop()
        assert self.curBlock != None, "too many block pops"
        self.builder.insertAtEnd(self.curBlock)

    def handleRetStat(self, ctx, retstat):
        vals = []
        tail = []
        if retstat and retstat.explist():
            vals, tail = self.getValsAndTail(retstat.explist())
        makeConcat(self.builder, vals, tail, self.getEndLoc(ctx),
                   concatLike=lua.ret)

    def handleCondRetstat(self, ctx, retstat):
        if retstat:
            self.handleRetStat(ctx, retstat)
        else:
            self.builder.create(lua.end, loc=self.getEndLoc(ctx))

    def getValsAndTail(self, ctx):
        vals = []
        for i in range(0, len(ctx.exp())):
            vals.append(self.exp(ctx.exp(i),
                                 allowPack=((i+1)==len(ctx.exp()))))
        tail = []
        if len(vals) > 0 and vals[-1].type == lua.pack():
            tail = [vals[-1]]
            vals = vals[0:-1]
        return vals, tail

    def handleAssignList(self, varList, explist):
        expPack = self.explist(explist)
        vals = self.builder.create(
                lua.unpack, pack=expPack, vals=[lua.val()] * len(varList),
                loc=self.getStartLoc(explist)).vals()
        for i in range(0, len(varList)):
            self.builder.create(lua.assign, tgt=varList[i], val=vals[i],
                                loc=self.getStartLoc(explist))

    #########################################################
    # AST Walker                                            #
    #########################################################

    def chunk(self, ctx:LuaParser.ChunkContext):
        # Create the Lua module
        module = ModuleOp()
        main = FuncOp("lua_main", FunctionType([], [lua.pack()]), self.getStartLoc(ctx))
        module.append(main)

        # Set the current region
        self.region = main.getBody()
        self.pushBlock(self.region.addEntryBlock([]))

        # Process the block
        self.block(ctx.block())
        assert ctx.block().retstat() == None, "unexpected return statement"
        makeConcat(self.builder, [], [], self.getEndLoc(ctx),
                   concatLike=lua.ret)

        # Return the module
        assert verify(module), "module failed to verify"
        return module, main

    def block(self, ctx:LuaParser.BlockContext):
        # Process the statements
        for stat in ctx.stat():
            self.stat(stat)

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
            self.whileloop(ctx.whileloop())
        elif ctx.repeatloop():
            self.repeatloop(ctx.repeatloop())
        elif ctx.conditionalchain():
            self.conditionalchain(ctx.conditionalchain())
        elif ctx.numericfor():
            self.numericfor(ctx.numericfor())
        elif ctx.genericfor():
            self.genericfor(ctx.genericfor())
        elif ctx.namedfunctiondef():
            self.namedfunctiondef(ctx.namedfunctiondef())
        elif ctx.localnamedfunctiondef():
            self.localnamedfunctiondef(ctx.localnamedfunctiondef())
        elif ctx.localvarlist():
            self.localvarlist(ctx.localvarlist())
        elif ctx.getText() == ";":
            pass
        else:
            raise ValueError("Unknown StatContext case")

    def assignlist(self, ctx:LuaParser.AssignlistContext):
        varList = self.varlist(ctx.varlist())
        self.handleAssignList(varList, ctx.explist())

    def localvarlist(self, ctx:LuaParser.LocalvarlistContext):
        varList = []
        loc = self.getStartLoc(ctx)
        for varName in ctx.namelist().NAME():
            lalloc = self.builder.create(lua.alloc_local, loc=loc,
                                         var=StringAttr(varName.getText()))
            varList.append(lalloc.res())

        if ctx.explist():
            self.handleAssignList(varList, ctx.explist())

    def varlist(self, ctx:LuaParser.VarlistContext):
        varList = []
        for var in ctx.var():
            varList.append(self.var(var))
        return varList

    def explist(self, ctx:LuaParser.ExplistContext):
        vals, tail = self.getValsAndTail(ctx)
        return makeConcat(self.builder, vals, tail, self.getStartLoc(ctx)).pack()

    def nameAndArgs(self, ctx:LuaParser.NameAndArgsContext):
        assert ctx.NAME() == None, "colon operator unimplemented"
        return self.args(ctx.args())

    def args(self, ctx:LuaParser.ArgsContext):
        if ctx.explist():
            return self.explist(ctx.explist())
        vals = []
        if ctx.tableconstructor():
            # Ensure return value is a list of values
            vals = [self.tableconstructor(ctx.tableconstructor())]
        elif ctx.string():
            vals = [self.string(ctx.string())]
        return makeConcat(self.builder, vals, [], self.getStartLoc(ctx)).pack()

    def varOrExp(self, ctx:LuaParser.VarOrExpContext):
        if ctx.var():
            return self.var(ctx.var())
        elif ctx.exp():
            return self.exp(ctx.exp())
        else:
            raise ValueError("Unknown VarOrExpContext case")

    def var(self, ctx:LuaParser.VarContext):
        assert ctx.exprVar() == None, "expression variables unsupported"
        val = self.builder.create(lua.get_or_alloc,
                                  var=StringAttr(ctx.NAME().getText()),
                                  loc=self.getStartLoc(ctx)).res()
        for suffix in ctx.varSuffix():
            val = self.varSuffix(val, suffix)
        return val

    def varSuffix(self, var, ctx:LuaParser.VarSuffixContext):
        val = self.handleFunctionLike(var, ctx, unpackLast=True)
        loc = self.getStartLoc(ctx)
        if ctx.exp():
            key = self.exp(ctx.exp())
        else:
            key = self.builder.create(lua.get_string,
                                      value=StringAttr(ctx.NAME().getText()),
                                      loc=loc).res()
        return self.builder.create(lua.table_get, tbl=val, key=key, loc=loc).val()

    def exp(self, ctx:LuaParser.ExpContext, allowPack=False):
        if ctx.nilvalue():
            return self.nilvalue(ctx.nilvalue())
        elif ctx.falsevalue():
            return self.falsevalue(ctx.falsevalue())
        elif ctx.truevalue():
            return self.truevalue(ctx.truevalue())
        elif ctx.number():
            return self.number(ctx.number())
        elif ctx.string():
            return self.string(ctx.string())
        elif ctx.elipsis():
            raise NotImplementedError("elipsis not implemented")
        elif ctx.functiondef():
            return self.functiondef(ctx.functiondef())
        elif ctx.prefixexp():
            return self.prefixexp(ctx.prefixexp(), allowPack=allowPack)
        elif ctx.tableconstructor():
            return self.tableconstructor(ctx.tableconstructor())
        elif ctx.operatorUnary():
            return self.operatorUnary(ctx.exp(0), ctx.operatorUnary())
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

    def operatorUnary(self, exp, ctx:LuaParser.OperatorUnaryContext):
        unary = self.builder.create(lua.unary, op=StringAttr(ctx.getText()),
                                    val=self.exp(exp), loc=self.getStartLoc(ctx))
        return unary.res()

    def nilvalue(self, ctx:LuaParser.NilvalueContext):
        nil = self.builder.create(lua.nil, loc=self.getStartLoc(ctx))
        return nil.res()

    def falsevalue(self, ctx):
        return self.handleBool(self.getStartLoc(ctx), 0)

    def truevalue(self, ctx):
        return self.handleBool(self.getStartLoc(ctx), 1)

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

    def string(self, ctx:LuaParser.StringContext):
        # TODO escapes and special characters unhandled
        if ctx.NORMALSTRING():
            text = ctx.NORMALSTRING().getText()
        elif ctx.CHARSTRING():
            text = ctx.CHARSTRING().getText()
        elif ctx.LONGSTRING():
            raise NotImplementedError("long strings not implemented")
        else:
            raise ValueError("Unknown StringContext case")
        text = text[1:len(text)-1]
        return self.builder.create(lua.get_string, value=StringAttr(text),
                                   loc=self.getStartLoc(ctx)).res()

    def tableconstructor(self, ctx:LuaParser.TableconstructorContext):
        loc = self.getStartLoc(ctx)
        tbl = self.builder.create(lua.table, loc=loc).res()

        idx = 1
        if not ctx.fieldlist():
            return tbl
        for field in ctx.fieldlist().field():
            floc = self.getStartLoc(field)
            if len(field.exp()) == 2:
                key = self.exp(field.exp(0))
                val = self.exp(field.exp(1))
            elif field.NAME() != None:
                keyStr = StringAttr(field.NAME().getText())
                key = self.builder.create(lua.get_string, value=keyStr, loc=floc).res()
                val = self.exp(field.exp(0))
            else:
                key = self.builder.create(lua.number, value=F64Attr(idx), loc=floc).res()
                val = self.exp(field.exp(0))
                idx += 1
            self.builder.create(lua.table_set, tbl=tbl, key=key, val=val, loc=floc)
        return tbl

    def prefixexp(self, ctx:LuaParser.PrefixexpContext, allowPack):
        # A variable or expression, with optional trailing function calls
        # `myFcn (a, b, c) {a: b} "abc" {} "" (c, b, d)` is valid syntax
        return self.handleFunctionLike(self.varOrExp(ctx.varOrExp()), ctx,
                                       unpackLast=not allowPack)

    def functioncall(self, ctx:LuaParser.FunctioncallContext):
        # Identical to prefixexp, except len(allArgs) >= 1, and statement does
        # not return a value
        self.prefixexp(ctx, allowPack=True)

    #########################################################
    # Control-Flow Instructions                             #
    #########################################################

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
        self.pushBlock(loop.region().addEntryBlock([lua.val()]))
        # Process the block and restore builder
        self.block(ctx.block())
        assert ctx.block().retstat() == None, "unsupported return statement"
        self.builder.create(lua.end, loc=loc)
        self.popBlock()

    def funcname(self, ctx:LuaParser.FuncnameContext):
        assert len(ctx.NAME()) == 1, "only simple function names supported"
        return StringAttr(ctx.NAME(0).getText())

    def parlist(self, ctx:LuaParser.ParlistContext):
        assert ctx.elipsis() == None, "variadic functions unsupported"
        return ArrayAttr([StringAttr(n.getText()) for n in ctx.namelist().NAME()])

    def funcbody(self, ctx:LuaParser.FuncbodyContext):
        return self.parlist(ctx.parlist()) if ctx.parlist() else ArrayAttr([])

    def functiondef(self, ctx:LuaParser.FunctiondefContext):
        params = self.funcbody(ctx.funcbody())
        loc = self.getStartLoc(ctx)
        fcnDef = self.builder.create(lua.function_def, params=params, loc=loc)

        self.pushBlock(fcnDef.region().addEntryBlock([lua.val()] * len(params)))
        self.block(ctx.funcbody().block())
        retstat = ctx.funcbody().block().retstat()
        self.handleRetStat(ctx, ctx.funcbody().block().retstat())
        self.popBlock()
        return fcnDef.fcn()

    def localnamedfunctiondef(self, ctx):
        fcn = self.functiondef(ctx)
        var = self.builder.create(lua.alloc_local, loc=self.getStartLoc(ctx),
                                  var=StringAttr(ctx.NAME().getText())).res()
        self.builder.create(lua.assign, tgt=var, val=fcn,
                            loc=self.getStartLoc(ctx))

    def namedfunctiondef(self, ctx:LuaParser.NamedfunctiondefContext):
        # same as functiondef except with assignment to named variable
        fcn = self.functiondef(ctx)
        name = self.funcname(ctx.funcname())
        alloc = self.builder.create(lua.get_or_alloc, var=name,
                                    loc=self.getStartLoc(ctx))
        self.builder.create(lua.assign, tgt=alloc.res(), val=fcn,
                            loc=self.getStartLoc(ctx))

    def ifblock(self, ctx):
        cond = self.exp(ctx.exp())
        cond_if = self.builder.create(lua.cond_if, cond=cond,
                                      loc=self.getStartLoc(ctx))
        self.pushBlock(cond_if.first().addEntryBlock([]))
        self.block(ctx.block())
        self.handleCondRetstat(ctx, ctx.block().retstat())
        self.popBlock()

        self.pushBlock(cond_if.second().addEntryBlock([]))

    def elseblock(self, ctx):
        self.block(ctx.block())
        self.handleCondRetstat(ctx, ctx.block().retstat())

    def conditionalchain(self, ctx:LuaParser.ConditionalchainContext):
        self.ifblock(ctx.ifblock())
        for elseif in ctx.elseifblock():
            self.ifblock(elseif)
        if ctx.elseblock():
            self.elseblock(ctx.elseblock())
        else:
            self.builder.create(lua.end, loc=self.getEndLoc(ctx.ifblock()))
        for elseif in ctx.elseifblock():
            self.popBlock()
            self.builder.create(lua.end, loc=self.getEndLoc(elseif))
        self.popBlock()

    def whileloop(self, ctx:LuaParser.WhileloopContext):
        loop = self.builder.create(lua.loop_while, loc=self.getStartLoc(ctx))
        self.pushBlock(loop.eval().addEntryBlock([]))
        cond = self.exp(ctx.exp())
        self.builder.create(lua.cond, cond=cond, loc=self.getStartLoc(ctx.exp()))
        self.popBlock()
        self.pushBlock(loop.region().addEntryBlock([]))
        self.block(ctx.block())
        assert ctx.block().retstat() == None, "unexpected terminator in while"
        self.builder.create(lua.end, loc=self.getEndLoc(ctx.block()))
        self.popBlock()

    def repeatloop(self, ctx:LuaParser.RepeatloopContext):
        loop = self.builder.create(lua.repeat, loc=self.getStartLoc(ctx))
        self.pushBlock(loop.region().addEntryBlock([]))
        self.block(ctx.block())
        assert ctx.block().retstat() == None, "unexpected terminator in repeat"
        until = self.builder.create(lua.until, loc=self.getStartLoc(ctx.exp()))
        self.pushBlock(until.eval().addEntryBlock([]))
        cond = self.exp(ctx.exp())
        self.builder.create(lua.cond, cond=cond, loc=self.getStartLoc(ctx.exp()))
        self.popBlock()
        self.popBlock()

    def genericfor(self, ctx:LuaParser.GenericforContext):
        loc = self.getStartLoc(ctx)
        params = ArrayAttr([StringAttr(t.getText()) for t in ctx.namelist().NAME()])
        expPack = self.explist(ctx.explist())
        itVars = self.builder.create(lua.unpack, pack=expPack,
                                     vals=[lua.val()] * 3, loc=loc).vals()
        loop = self.builder.create(lua.generic_for, f=itVars[0], s=itVars[1],
                                   var=itVars[2], params=params, loc=loc)
        self.pushBlock(loop.region().addEntryBlock([lua.val()] * len(params)))
        self.block(ctx.block())
        assert ctx.block().retstat() == None, "unexpected terminator statement"
        self.builder.create(lua.end, loc=self.getEndLoc(ctx))
        self.popBlock()

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
        raise ValueError("failed to lookup variable")

    def set_local(self, k, e):
        self.scopes[-1][k] = e

    def set_global(self, k, e):
        self.scopes[0][k] = e

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
        self.worklist = []

    def addRegion(self, region):
        for o in region.getBlock(0):
            self.checkWork(o)

    def addWork(self, op): self.worklist.append(op)
    def pushScope(self): self.addWork("push_scope")
    def popScope(self): self.addWork("pop_scope")

    def checkWork(self, op):
        if isa(op, FuncOp):
            self.addRegion(op.getRegion(0))
        elif isa(op, lua.numeric_for) or isa(op, lua.generic_for) or isa(op, lua.function_def):
            self.pushScope()
            self.addWork(op)
            self.addRegion(op.getRegion(0))
            self.popScope()
        else:
            self.addWork(op)
            for r in op.getRegions():
                self.pushScope()
                self.addRegion(r)
                self.popScope()

    def visitAll(self, main:FuncOp):
        self.global_block = main.getBody().getBlock(0)
        self.checkWork(main)
        for op in self.worklist:
            if op not in self.removed:
                self.visit(op)

    def visit(self, op:Operation):
        if op == "push_scope":
            self.scope.push_scope()
            return
        if op == "pop_scope":
            self.scope.pop_scope()
            return

        elif isa(op, lua.get_or_alloc):
            self.visitGetOrAlloc(lua.get_or_alloc(op))
        elif isa(op, lua.alloc_local):
            self.visitAllocLocal(lua.alloc_local(op))
        elif isa(op, lua.numeric_for):
            self.visitNumericFor(lua.numeric_for(op))
        elif isa(op, lua.generic_for):
            self.visitGenericFor(lua.generic_for(op))
        elif isa(op, lua.function_def):
            self.visitFunctionDef(lua.function_def(op))
        elif isa(op, lua.assign):
            self.visitAssign(lua.assign(op))

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

    def visitGenericFor(self, op:lua.generic_for):
        for i in range(0, len(op.params())):
            self.scope.set_local(op.params()[i],
                                 op.region().getBlock(0).getArgument(i))

    def visitFunctionDef(self, op:lua.function_def):
        for i in range(0, len(op.params())):
            self.scope.set_local(op.params()[i],
                                 op.region().getBlock(0).getArgument(i))

    def visitAssign(self, op):
        if (not isa(op.tgt().definingOp, lua.alloc) or
                not op.val().hasOneUse() or
                op.tgt().definingOp.parentOp != op.val().definingOp.parentOp):
            return
        self.scope.set_local(lua.alloc(op.tgt().definingOp).var(), op.res())

################################################################################
# High-Level IR: Basic Optimizations and Capture Handling
################################################################################

def explicitCapture(op:lua.function_def, rewriter:Builder):
    captures = set()
    def collectCaptures(o):
        for val in o.getOperands():
            if ((val.definingOp and not op.isProperAncestor(val.definingOp)) or (
                 hasattr(val.owner, "parent") and
                 val.owner.parent.isProperAncestor(op.region()))):
                captures.add(val)
    walkInOrder(op, collectCaptures)
    captures = list(captures)
    fcnDef = rewriter.create(lua.function_def_capture, captures=captures,
                             params=op.params(), loc=op.loc)
    prev = op.region().getBlock(0)
    entry = fcnDef.region().addEntryBlock(
        [lua.val()] * (len(captures) + prev.getNumArguments()))
    bvm = BlockAndValueMapping()
    for i in range(0, len(captures)):
        bvm[captures[i]] = entry.getArgument(i)
    for i in range(0, prev.getNumArguments()):
        bvm[prev.getArgument(i)] = entry.getArgument(len(captures) + i)
    copyInto(entry, op.region().getBlock(0), None, bvm)
    rewriter.replace(op, [fcnDef.fcn()])
    return True

def elideConcatAndUnpack(op:lua.unpack, rewriter:Builder):
    parent = op.pack().definingOp
    if not isa(parent, lua.concat):
        return False
    concat = lua.concat(parent)
    newVals = []
    for i in range(0, min(len(concat.vals()), len(op.vals()))):
        newVals.append(concat.vals()[i])
    if len(concat.tail()) == 0:
        nil = rewriter.create(lua.nil, loc=op.loc)
        for i in range(len(newVals), len(op.vals())):
            newVals.append(nil.res())
    else:
        nVals = len(op.vals()) - len(newVals)
        newVals += rewriter.create(lua.unpack, pack=concat.tail()[0],
                                   vals=[lua.val()] * nVals, loc=op.loc).vals()
    rewriter.replace(op, newVals)
    rewriter.erase(concat)
    return True

def elideConcatPack(op, rewriter):
    if len(op.vals()) != 0 or len(op.tail()) == 0:
        return False
    rewriter.replace(op, op.tail())
    return True

lua_builtins = set(["print", "string", "io", "table", "math"])
def raiseBuiltins(op:lua.alloc, rewriter:Builder):
    if op.var().getValue() not in lua_builtins:
        return False
    builtin = rewriter.create(lua.builtin, var=op.var(), loc=op.loc)
    rewriter.replace(op, [builtin.val()])
    return True

def assignTableSet(op:lua.assign, rewriter:Builder):
    if not isa(op.tgt().definingOp, lua.table_get):
        return False
    tableGet = lua.table_get(op.tgt().definingOp)
    assert tableGet.val().hasOneUse()
    rewriter.create(lua.table_set, tbl=tableGet.tbl(), key=tableGet.key(),
                    val=op.val(), loc=tableGet.loc)
    rewriter.erase(op)
    rewriter.erase(tableGet)
    return False

def isCaptured(val):
    return any(isa(use, lua.function_def_capture) for use in val.getOpUses())

def elideAssign(op, rewriter):
    if (not isa(op.tgt().definingOp, lua.alloc) or
            not op.val().hasOneUse() or
            op.tgt().definingOp.parentOp != op.val().definingOp.parentOp or
            isCaptured(op.tgt())):
        rewriter.create(lua.copy, tgt=op.tgt(), val=op.val(), loc=op.loc)
        rewriter.replace(op, [op.tgt()])
    else:
        rewriter.replace(op, [op.val()])
    return True

def constNumber(op, rewriter):
    if not neverWrittenTo(op.res()):
        return False
    const = rewriter.create(luaopt.const_number, value=op.value(), loc=op.loc)
    rewriter.replace(op, [const.res()])
    return True

def varAllocPass(module, main:FuncOp):
    # Non-trivial passes
    AllocVisitor().visitAll(main)

    applyOptPatterns(main, [Pattern(lua.function_def, explicitCapture)])
    applyOptPatterns(main, [
        Pattern(lua.unpack, elideConcatAndUnpack, [lua.nil]),
        Pattern(lua.concat, elideConcatPack),
        Pattern(lua.alloc, raiseBuiltins, [lua.builtin]),
        Pattern(lua.assign, assignTableSet),
    ])
    applyOptPatterns(main, [Pattern(lua.assign, elideAssign)])
    applyOptPatterns(main, [Pattern(lua.number, constNumber)])
    #applyLICM(module)
    applyCSE(module, licmCanHoist)

################################################################################
# IR: Dialect Conversion to SCF
################################################################################

def copyInto(newBlk, oldBlk, termCls=None, bvm=None):
    if not bvm:
        bvm = BlockAndValueMapping()
    for oldOp in oldBlk:
        if not termCls or not isa(oldOp, termCls):
            newBlk.append(oldOp.clone(bvm))

def argPackFunctionDef(op, rewriter):
    packFcn = rewriter.create(luaopt.pack_func, captures=list(op.captures()),
                              loc=op.loc)
    entry = packFcn.region().addEntryBlock([lua.capture(), lua.pack()])
    rewriter.insertAtStart(entry)
    caps = rewriter.create(lua.get_captures, capture=entry.getArgument(0),
                           vals=[lua.val()] * len(op.captures()), loc=op.loc)
    args = rewriter.create(lua.unpack, pack=entry.getArgument(1),
                           vals=[lua.val()] * len(op.params()), loc=op.loc)
    bvm = BlockAndValueMapping()
    prev = op.region().getBlock(0)
    for i in range(0, len(op.captures())):
        bvm[prev.getArgument(i)] = caps.vals()[i]
    for i in range(0, len(op.params())):
        bvm[prev.getArgument(len(op.captures()) + i)] = args.vals()[i]
    copyInto(entry, prev, None, bvm)
    rewriter.replace(op, [packFcn.fcn()])
    return True

def lowerNumericFor(op:lua.numeric_for, rewriter:Builder):
    step = rewriter.create(luac.get_int_val, val=op.step(), loc=op.loc).num()
    lower = rewriter.create(luac.get_int_val, val=op.lower(), loc=op.loc)
    i = rewriter.create(luac.wrap_int, num=lower.num(), loc=op.loc).res()
    loopWhile = rewriter.create(lua.loop_while, loc=op.loc)
    rewriter.insertAtStart(loopWhile.eval().addEntryBlock([]))
    le = rewriter.create(luac.le, lhs=i, rhs=op.upper(), loc=op.loc)
    rewriter.create(lua.cond, cond=le.res(), loc=op.loc)

    bvm = BlockAndValueMapping()
    bvm[op.region().getBlock(0).getArgument(0)] = i
    copyInto(loopWhile.region().addEntryBlock([]), op.region().getBlock(0),
             None, bvm)

    rewriter.insertBefore(loopWhile.region().getBlock(0).getTerminator())
    iv = rewriter.create(luac.get_int_val, val=i, loc=op.loc).num()
    newI = rewriter.create(AddIOp, lhs=iv, rhs=step, ty=I64Type(), loc=op.loc).result()
    var = rewriter.create(luac.wrap_int, num=newI, loc=op.loc).res()
    rewriter.create(lua.copy, tgt=i, val=var, loc=op.loc)
    rewriter.erase(op)
    return True

def lowerGenericFor(op:lua.generic_for, rewriter:Builder):
    fcnPack = makeConcat(rewriter, [op.s(), op.var()], [], op.loc)
    paramPack = rewriter.create(lua.call, fcn=op.f(), args=fcnPack.pack(),
                                loc=op.loc).rets()
    nil = rewriter.create(lua.nil, loc=op.loc).res()
    params = rewriter.create(lua.unpack, pack=paramPack, loc=op.loc,
                             vals=[lua.val()] * len(op.params())).vals()
    loopWhile = rewriter.create(lua.loop_while, loc=op.loc)
    rewriter.insertAtStart(loopWhile.eval().addEntryBlock([]))
    rewriter.create(lua.copy, tgt=op.var(), val=params[0], loc=op.loc)
    ne = rewriter.create(luac.ne, lhs=op.var(), rhs=nil, loc=op.loc)
    rewriter.create(lua.cond, cond=ne.res(), loc=op.loc)

    bvm = BlockAndValueMapping()
    for i in range(0, len(op.params())):
        bvm[op.region().getBlock(0).getArgument(i)] = params[i]
    copyInto(loopWhile.region().addEntryBlock([]), op.region().getBlock(0),
             None, bvm)

    rewriter.insertBefore(loopWhile.region().getBlock(0).getTerminator())
    nextFcnPack = makeConcat(rewriter, [op.s(), op.var()], [], loc=op.loc)
    nextPack = rewriter.create(lua.call, fcn=op.f(), args=nextFcnPack.pack(),
                               loc=op.loc).rets()
    nextParams = rewriter.create(lua.unpack, pack=nextPack, loc=op.loc,
                                 vals=[lua.val()] * len(op.params())).vals()
    for i in range(0, len(nextParams)):
        rewriter.create(lua.copy, tgt=params[i], val=nextParams[i], loc=op.loc)
    rewriter.erase(op)
    return True

def capturesSelf(op, cap):
    for use in cap.getOpUses():
        if not isa(use, lua.copy):
            continue
        if op.fcn() == lua.copy(use).val():
            return True
    return False

def handleCondTerm(term:Operation, after:Block, rewriter:Builder):
    if isa(term, lua.end):
        rewriter.insertBefore(term)
        rewriter.create(BranchOp, dest=after, loc=term.loc)
        rewriter.erase(term)
    elif isa(term, lua.ret):
        return
    else:
        raise ValueError("expected lua.ret or lua.end as terminator")

def lowerCondIf(op:lua.cond_if, rewriter:Builder):
    boolVal = rewriter.create(luac.convert_bool_like, val=op.cond(), loc=op.loc).b()
    # Add the conditional blocks [before, first, second, after]
    before = op.block
    after = before.split(op)
    first = Block()
    second = Block()
    second.insertBefore(after)
    first.insertBefore(second)
    # cond_br
    rewriter.insertAtEnd(before)
    rewriter.create(CondBranchOp, loc=op.loc, cond=boolVal, trueDest=first,
                    falseDest=second)
    # trueDest
    copyInto(first, op.first().getBlock(0))
    handleCondTerm(first.getTerminator(), after, rewriter)
    # falseDest
    copyInto(second, op.second().getBlock(0))
    handleCondTerm(second.getTerminator(), after, rewriter)
    rewriter.erase(op)
    return True

def lowerLoopWhile(op:lua.loop_while, rewriter:Builder):
    before = op.block
    after = before.split(op)
    cond = Block()
    body = Block()
    body.insertBefore(after)
    cond.insertBefore(body)
    # br to eval
    rewriter.insertAtEnd(before)
    rewriter.create(BranchOp, dest=cond, loc=op.loc)
    # cond
    copyInto(cond, op.eval().getBlock(0))
    condTerm = lua.cond(cond.getTerminator())
    rewriter.insertBefore(condTerm)
    condBool = rewriter.create(luac.convert_bool_like, val=condTerm.cond(),
                               loc=condTerm.loc).b()
    rewriter.create(CondBranchOp, cond=condBool, trueDest=body, falseDest=after,
                    loc=op.loc)
    rewriter.erase(condTerm)
    # body
    copyInto(body, op.region().getBlock(0))
    bodyTerm = lua.end(body.getTerminator())
    rewriter.insertBefore(bodyTerm)
    rewriter.create(BranchOp, dest=cond, loc=bodyTerm.loc)
    rewriter.erase(bodyTerm)
    rewriter.erase(op)
    return True

def lowerRepeatUntil(op:lua.until, rewriter:Builder):
    assert isa(op.parentOp, lua.repeat), "expected lua.repeat as parent"
    repeat = lua.repeat(op.parentOp)
    before = repeat.block
    after = before.split(repeat)
    body = Block()
    body.insertBefore(after)
    # br to body
    rewriter.insertAtEnd(before)
    rewriter.create(BranchOp, dest=body, loc=repeat.loc)
    # body
    copyInto(body, repeat.region().getBlock(0), lua.until)
    # cond
    copyInto(body, op.eval().getBlock(0))
    condTerm = lua.cond(body.getTerminator())
    rewriter.insertBefore(condTerm)
    condBool = rewriter.create(luac.convert_bool_like, val=condTerm.cond(),
                               loc=condTerm.loc).b()
    rewriter.create(CondBranchOp, cond=condBool, trueDest=after, falseDest=body,
                    loc=op.loc)
    rewriter.erase(condTerm)
    rewriter.erase(repeat)
    return True

def getFcnDef(val):
    if isa(val.definingOp, luaopt.pack_func):
        return luaopt.pack_func(val.definingOp)
    if isa(val.definingOp, lua.alloc):
        for use in val.getOpUses():
            if isa(use, lua.copy):
                return getFcnDef(lua.assign(use).val())
    if isa(val.definingOp, lua.get_captures):
        cap = lua.get_captures(val.definingOp).capture()
        return luaopt.pack_func(cap.owner.parent.parentOp)
    return None

def knownCallUnpack(op, rewriter):
    if not isa(op.pack().definingOp, lua.call): return False
    fcnVal = lua.call(op.pack().definingOp).fcn()
    fcnDef = getFcnDef(fcnVal)
    if not fcnDef: return False
    minSz = None
    for bb in fcnDef.region():
        term = bb.getTerminator()
        if not isa(term, lua.ret): continue
        nVals = len(lua.ret(term).vals())
        minSz = min(minSz, nVals) if minSz != None else nVals
    assert minSz != None
    if minSz < len(op.vals()): return False
    vals = rewriter.create(luaopt.unpack_unsafe, pack=op.pack(),
                           vals=[lua.val()] * len(op.vals()), loc=op.loc).vals()
    rewriter.replace(op, vals)
    return True

anon_name_counter = 0
def lowerFunctionDef(module):
    def lowerFcn(op, rewriter):
        caps = rewriter.create(lua.make_capture, vals=list(op.captures()),
                               loc=op.loc).capture()

        global anon_name_counter
        name = "lua_anon_fcn_" + str(anon_name_counter)
        anon_name_counter += 1
        func = FuncOp(name, luac.pack_fcn(), op.loc)
        module.append(func)
        func.getBody().takeBody(op.region())
        fcnAddr = rewriter.create(ConstantOp, value=FlatSymbolRefAttr(name),
                                  ty=luac.pack_fcn(), loc=op.loc).result()
        fcn = rewriter.create(luac.make_fcn, addr=fcnAddr, capture=caps,
                              loc=op.loc).fcn()
        rewriter.replace(op, [fcn])
        return True
    return lowerFcn

def cfExpand(module:ModuleOp, main:FuncOp):
    applyOptPatterns(module, [Pattern(lua.function_def_capture,
                                      argPackFunctionDef)])
    applyOptPatterns(module, [
        Pattern(lua.numeric_for, lowerNumericFor),
        Pattern(lua.generic_for, lowerGenericFor),
        Pattern(lua.loop_while, lowerLoopWhile),
        Pattern(lua.until, lowerRepeatUntil),
        Pattern(lua.cond_if, lowerCondIf),
    ])
    applyOptPatterns(module, [Pattern(lua.unpack, knownCallUnpack)])
    applyOptPatterns(module, [Pattern(luaopt.pack_func,
                                      lowerFunctionDef(module))])

################################################################################
# IR: Optimizations
################################################################################

def preallocValid(op, rewriter):
    numOp = op.key().definingOp
    if not isa(numOp, luaopt.const_number):
        return None
    value = luaopt.const_number(numOp).value()
    if hasattr(value, "getValue"):
        raw = value.getValue()
    else:
        raw = value.getInt()
    ivVal = int(raw)
    if ivVal != raw or ivVal <= 0 or ivVal > luaopt.table_prealloc().getInt():
        return None
    #return rewriter.create(ConstantOp, value=I64Attr(ivVal - 1),
    #                       loc=op.loc).result()
    return rewriter.create(ConstantOp, value=I64Attr(ivVal - 1),
                           loc=op.loc).result()

def tableGetPrealloc(op:lua.table_get, rewriter:Builder):
    iv = preallocValid(op, rewriter)
    if iv == None:
        return False
    val = rewriter.create(luaopt.table_get_prealloc, tbl=op.tbl(), iv=iv,
                          loc=op.loc).val()
    rewriter.replace(op, [val])
    return True

def tableSetPrealloc(op:lua.table_set, rewriter:Builder):
    iv = preallocValid(op, rewriter)
    if iv == None:
        return False
    rewriter.create(luaopt.table_set_prealloc, tbl=op.tbl(), iv=iv,
                    val=op.val(), loc=op.loc)
    rewriter.erase(op)
    return True

def applyOpts(module):
    applyOptPatterns(module, [
        Pattern(lua.table_get, tableGetPrealloc),
        Pattern(lua.table_set, tableSetPrealloc),
    ])
    applyCSE(module, licmCanHoist)

################################################################################
# IR: Dialect Conversion to StandardOps and LuaC Dialect
################################################################################

def lowerAlloc(op:lua.alloc, rewriter:Builder):
    nil = rewriter.create(lua.nil, loc=op.loc)
    rewriter.replace(op, [nil.res()])
    return True

def luaBooleanWrap(op, b):
    const = b.create(ConstantOp, value=op.value(), loc=op.loc).result()
    val = b.create(luac.wrap_bool, b=const, loc=op.loc).res()
    b.replace(op, [val])
    return True

def luaNumberWrap(op, rewriter):
    num = rewriter.create(ConstantOp, value=op.value(), loc=op.loc).result()
    if num.type == F64Type():
        val = rewriter.create(luac.wrap_real, num=num, loc=op.loc).res()
    else:
        val = rewriter.create(luac.wrap_int, num=num, loc=op.loc).res()
    rewriter.replace(op, [val])
    return True

def getExpanderFor(opStr, binOpCls):
    def expandFcn(op:lua.binary, rewriter:Builder):
        if op.op().getValue() != opStr:
            return False
        binOp = rewriter.create(binOpCls, lhs=op.lhs(), rhs=op.rhs(),
                                loc=op.loc)
        rewriter.replace(op, [binOp.res()])
        return True

    return expandFcn

def getUnaryExpander(opStr, unOpCls):
    def expandFcn(op:lua.unary, rewriter:Builder):
        if op.op().getValue() != opStr:
            return False
        unOp = rewriter.create(unOpCls, val=op.val(), loc=op.loc)
        rewriter.replace(op, [unOp.res()])
        return True

    return expandFcn

def expandConcat(op:lua.concat, rewriter:Builder):
    assert op.pack().hasOneUse(), "value pack can only be used once"
    getter = (luac.get_ret_pack if isa(op.pack().getOpUses()[0], ReturnOp) else
              luac.get_arg_pack)
    szVar = rewriter.create(ConstantOp, value=I32Attr(len(op.vals())),
                            loc=op.loc).result()
    tail = None if len(op.tail()) == 0 else op.tail()[0]
    if tail:
        tailSz = rewriter.create(luac.pack_get_size, pack=tail, size=I32Type(),
                                 loc=op.loc).size()
        szVar = rewriter.create(AddIOp, lhs=szVar, rhs=tailSz, ty=I32Type(),
                                loc=op.loc).result()
    pack = rewriter.create(getter, size=szVar, loc=op.loc).pack()
    for i in range(0, len(op.vals())):
        idx = rewriter.create(ConstantOp, value=I32Attr(i), loc=op.loc).result()
        rewriter.create(luac.pack_insert, pack=pack, val=op.vals()[i], idx=idx,
                        loc=op.loc)
    if tail:
        idx = rewriter.create(ConstantOp, value=I32Attr(len(op.vals())),
                              loc=op.loc).result()
        rewriter.create(luac.pack_insert_all, pack=pack, tail=tail, idx=idx,
                        loc=op.loc)
    rewriter.replace(op, [pack])
    return True

def expandUnpack(op:lua.unpack, rewriter:Builder):
    newVals = []
    for i in range(0, len(op.vals())):
        idx = rewriter.create(ConstantOp, value=I32Attr(i), loc=op.loc).result()
        val = rewriter.create(luac.pack_get, pack=op.pack(), idx=idx,
                              loc=op.loc).res()
        newVals.append(val)
    rewriter.replace(op, newVals)
    return True

def expandUnpackUnsafe(op, rewriter):
    newVals = []
    for i in range(0, len(op.vals())):
        idx = rewriter.create(ConstantOp, value=I32Attr(i), loc=op.loc).result()
        val = rewriter.create(luac.pack_get_unsafe, pack=op.pack(), idx=idx,
                              loc=op.loc).res()
        newVals.append(val)
    rewriter.replace(op, newVals)
    return True

def expandCall(op:lua.call, rewriter:Builder):
    getAddr = rewriter.create(luac.get_fcn_addr, val=op.fcn(), loc=op.loc)
    getPack = rewriter.create(luac.get_capture_pack, val=op.fcn(), loc=op.loc)
    icall = rewriter.create(CallIndirectOp, callee=getAddr.addr(),
                            operands=[getPack.capture(), op.args()], loc=op.loc)
    rewriter.replace(op, icall.results())
    return True

def expandRet(op, rewriter):
    pack = makeConcat(rewriter, list(op.vals()), list(op.tail()), op.loc).pack()
    rewriter.create(ReturnOp, operands=[pack], loc=op.loc)
    rewriter.erase(op)
    return True

anon_string_counter = 0
def lowerGetString(module:ModuleOp):
    def lowerFcn(op:lua.get_string, rewriter:Builder):
        global anon_string_counter
        strName = StringAttr("lua_anon_string_" + str(anon_string_counter))
        anon_string_counter += 1
        module.append(luac.global_string(loc=op.loc, sym=strName,
                                         value=op.value()))
        loadStr = rewriter.create(luac.load_string, global_sym=strName,
                                  loc=op.loc)
        rewriter.replace(op, [loadStr.res()])
        return True

    return lowerFcn

def expandMakeCapture(op, rewriter):
    sz = rewriter.create(ConstantOp, value=I32Attr(len(op.vals())),
                         loc=op.loc).result()
    cap = rewriter.create(luac.new_capture, size=sz, loc=op.loc).capture()
    for i in range(0, len(op.vals())):
        idx = rewriter.create(ConstantOp, value=I32Attr(i), loc=op.loc).result()
        rewriter.create(luac.add_capture, capture=cap, val=op.vals()[i],
                        idx=idx, loc=op.loc)
    rewriter.replace(op, [cap])
    return True

def expandGetCaptures(op, rewriter):
    newVals = []
    for i in range(0, len(op.vals())):
        idx = rewriter.create(ConstantOp, value=I32Attr(i), loc=op.loc).result()
        val = rewriter.create(luac.get_capture, capture=op.capture(), idx=idx,
                              loc=op.loc).val()
        newVals.append(val)
    rewriter.replace(op, newVals)
    return True

def knownBool(op, rewriter):
    valid = [luac.eq, luac.ne, luac.lt, luac.le, luac.gt, luac.bool_and,
             luac.bool_not]
    if not any(isa(op.val().definingOp, opCls) for opCls in valid):
        return False
    b = rewriter.create(luac.get_bool_val, val=op.val(), loc=op.loc).b()
    rewriter.replace(op, [b])
    return True

def raiseConstNumber(op, rewriter):
    num = rewriter.create(lua.number, value=op.value(), loc=op.loc).res()
    rewriter.replace(op, [num])
    return True

def lowerToLuac(module:ModuleOp):
    target = ConversionTarget()
    target.addLegalDialect(luac)
    target.addLegalDialect(luaopt)
    target.addLegalDialect("std")
    target.addLegalDialect("scf")
    target.addLegalOp(FuncOp)
    target.addLegalOp(lua.table)
    target.addLegalOp(lua.nil)
    target.addLegalOp(lua.copy)
    target.addLegalOp(lua.builtin)
    target.addLegalOp(lua.table_get)
    target.addLegalOp(lua.table_set)
    target.addIllegalOp(luaopt.const_number)
    target.addIllegalOp(luaopt.unpack_unsafe)
    target.addLegalOp(ModuleOp)
    patterns = [
        Pattern(lua.alloc, lowerAlloc),
        Pattern(luaopt.const_number, raiseConstNumber),
        Pattern(lua.boolean, luaBooleanWrap),
        Pattern(lua.number, luaNumberWrap),

        Pattern(lua.binary, getExpanderFor("+", luac.add)),
        Pattern(lua.binary, getExpanderFor("-", luac.sub)),
        Pattern(lua.binary, getExpanderFor("*", luac.mul)),
        Pattern(lua.binary, getExpanderFor("^", luac.pow)),
        Pattern(lua.binary, getExpanderFor("..", luac.strcat)),
        Pattern(lua.binary, getExpanderFor("<", luac.lt)),
        Pattern(lua.binary, getExpanderFor(">", luac.gt)),
        Pattern(lua.binary, getExpanderFor("<=", luac.le)),
        Pattern(lua.binary, getExpanderFor(">=", luac.ge)),
        Pattern(lua.binary, getExpanderFor("==", luac.eq)),
        Pattern(lua.binary, getExpanderFor("~=", luac.ne)),
        Pattern(lua.binary, getExpanderFor("and", luac.bool_and)),

        Pattern(lua.unary, getUnaryExpander("not", luac.bool_not)),
        Pattern(lua.unary, getUnaryExpander("#", luac.list_size)),
        Pattern(lua.unary, getUnaryExpander("-", luac.neg), [luac.neg]),

        Pattern(lua.concat, expandConcat),
        Pattern(lua.unpack, expandUnpack),
        Pattern(luaopt.unpack_unsafe, expandUnpackUnsafe),
        Pattern(lua.make_capture, expandMakeCapture),
        Pattern(lua.get_captures, expandGetCaptures),
        Pattern(lua.call, expandCall),
        Pattern(lua.get_string, lowerGetString(module)),
        Pattern(lua.ret, expandRet),
    ]
    target.addLegalOp("module_terminator")
    applyFullConversion(module, patterns, target)
    applyOptPatterns(module, [Pattern(luac.convert_bool_like, knownBool)])

################################################################################
# IR: Lua to LLVMIR Pass 1
################################################################################

def llvmI32Const(b, iv, loc):
    return b.create(LLVMConstantOp, res=LLVMType.Int32(), value=I32Attr(iv),
                    loc=loc).res()

def llvmI64Const(b, iv, loc):
    return b.create(LLVMConstantOp, res=LLVMType.Int64(), value=I64Attr(iv),
                    loc=loc).res()

def allocaTyped(b, tyAttr, loc):
    ref = b.create(luallvm.alloca_value, loc=loc).ref()
    ty = b.create(luallvm.const_type, value=tyAttr, loc=loc).type()
    b.create(luallvm.set_type_direct, ref=ref, type=ty, loc=loc)
    return ref

def unpackTyAndU(b, ref, loc):
    return (b.create(luallvm.get_type_direct, ref=ref, loc=loc).type(),
            b.create(luallvm.get_u_direct, ref=ref, loc=loc).u())

def packTyAndU(b, ty, u, loc):
    undef = b.create(LLVMUndefOp, ty=luallvm.value(), loc=loc).res()
    val = b.create(LLVMInsertValueOp, res=luallvm.value(), container=undef,
                   value=ty, pos=I64ArrayAttr([0]), loc=loc).res()
    return b.create(LLVMInsertValueOp, res=luallvm.value(), container=val,
                    value=u, pos=I64ArrayAttr([1]), loc=loc).res()

def loadRef(b, ref, loc):
    return packTyAndU(b, *unpackTyAndU(b, ref, loc), loc)

def convertLuaNil(op, b):
    ref = allocaTyped(b, luac.type_nil(), op.loc)
    zeroI64 = llvmI64Const(b, 0, op.loc)
    b.create(luallvm.set_u_direct, ref=ref, u=zeroI64, loc=op.loc)
    b.replace(op, [ref])
    return True

def convertLuaTable(op, b):
    ref = allocaTyped(b, luac.type_tbl(), op.loc)
    impl = b.create(luallvm.new_table_impl, loc=op.loc).impl()
    b.create(luallvm.set_impl_direct, ref=ref, impl=impl, loc=op.loc)
    b.replace(op, [ref])
    return True

def convertLuacLoadString(op, b):
    ref = allocaTyped(b, luac.type_str(), op.loc)
    strData = b.create(luallvm.get_string_data, sym=op.global_sym(), loc=op.loc)
    impl = b.create(luallvm.load_string_impl, data=strData.data(),
                    length=strData.length(), loc=op.loc).impl()
    b.create(luallvm.set_impl_direct, ref=ref, impl=impl, loc=op.loc)
    b.replace(op, [ref])
    return True

def convertLuacWrapBool(op, b):
    ref = allocaTyped(b, luac.type_bool(), op.loc)
    # Zero-extend i1 to i64 and write directly to `ref.u`
    zextB = b.create(LLVMZExtOp, res=LLVMType.Int64(), value=op.b(),
                     loc=op.loc).res()
    b.create(luallvm.set_u_direct, ref=ref, u=zextB, loc=op.loc)
    b.replace(op, [ref])
    return True

def convertLuacWrapInt(op, b):
    ref = allocaTyped(b, luallvm.type_int(), op.loc)
    b.create(luallvm.set_u_direct, ref=ref, u=op.num(), loc=op.loc)
    b.replace(op, [ref])
    return True

def convertLuacWrapReal(op, b):
    ref = allocaTyped(b, luac.type_num(), op.loc)
    # Bitcast double to i64
    bcNum = b.create(LLVMBitcastOp, res=LLVMType.Int64(), arg=op.num(),
                     loc=op.loc).res()
    b.create(luallvm.set_u_direct, ref=ref, u=bcNum, loc=op.loc)
    b.replace(op, [ref])
    return True

def convertLuaCopy(op, b):
    ty = b.create(luallvm.get_type_direct, ref=op.val(), loc=op.loc).type()
    u = b.create(luallvm.get_u_direct, ref=op.val(), loc=op.loc).u()
    b.create(luallvm.set_type_direct, ref=op.tgt(), type=ty, loc=op.loc)
    b.create(luallvm.set_u_direct, ref=op.tgt(), u=u, loc=op.loc)
    b.erase(op)
    return True

def convertLuaTableGet(op, b):
    keyU = b.create(luallvm.get_u_direct, ref=op.key(), loc=op.loc).u()
    negOne = llvmI64Const(b, -1, op.loc)
    iv = b.create(AddIOp, ty=I64Type(), lhs=keyU, rhs=negOne, loc=op.loc).result()

    impl = b.create(luallvm.get_impl_direct, ref=op.tbl(), loc=op.loc).impl()
    arrPtr = b.create(LLVMBitcastOp, res=luallvm.ref(), arg=impl,
                      loc=op.loc).res()
    elPtr = b.create(LLVMGEPOp, res=luallvm.ref(), base=arrPtr,
                     indices=[iv], loc=op.loc).res()
    val = loadRef(b, elPtr, op.loc)

    #impl = b.create(luallvm.get_impl_direct, ref=op.tbl(), loc=op.loc).impl()
    #key = loadRef(b, op.key(), op.loc)
    #val = b.create(luallvm.table_get_impl, impl=impl, key=key, loc=op.loc).val()

    valPtr = b.create(luac.into_alloca, val=val, loc=op.loc).res()
    b.replace(op, [valPtr])
    return True

def convertLuaTableSet(op, b):
    keyU = b.create(luallvm.get_u_direct, ref=op.key(), loc=op.loc).u()
    negOne = llvmI64Const(b, -1, op.loc)
    iv = b.create(AddIOp, ty=I64Type(), lhs=keyU, rhs=negOne, loc=op.loc).result()

    impl = b.create(luallvm.get_impl_direct, ref=op.tbl(), loc=op.loc).impl()
    arrPtr = b.create(LLVMBitcastOp, res=luallvm.ref(), arg=impl,
                      loc=op.loc).res()
    elPtr = b.create(LLVMGEPOp, res=luallvm.ref(), base=arrPtr,
                     indices=[iv], loc=op.loc).res()
    ty, u = unpackTyAndU(b, op.val(), op.loc)
    b.create(luallvm.set_type_direct, ref=elPtr, type=ty, loc=op.loc)
    b.create(luallvm.set_u_direct, ref=elPtr, u=u, loc=op.loc)

    #impl = b.create(luallvm.get_impl_direct, ref=op.tbl(), loc=op.loc).impl()
    #key = loadRef(b, op.key(), op.loc)
    #val = loadRef(b, op.val(), op.loc)
    #b.create(luallvm.table_set_impl, impl=impl, key=key, val=val, loc=op.loc)

    b.erase(op)
    return True

def convertLuaoptTableGetPrealloc(op, b):
    impl = b.create(luallvm.get_impl_direct, ref=op.tbl(), loc=op.loc).impl()
    arrPtr = b.create(LLVMBitcastOp, res=luallvm.ref(), arg=impl,
                      loc=op.loc).res()
    elPtr = b.create(LLVMGEPOp, res=luallvm.ref(), base=arrPtr,
                     indices=[op.iv()], loc=op.loc).res()
    val = loadRef(b, elPtr, op.loc)

    #impl = b.create(luallvm.get_impl_direct, ref=op.tbl(), loc=op.loc).impl()
    #val = b.create(luallvm.table_get_prealloc_impl, impl=impl, iv=op.iv(),
    #               loc=op.loc).val()

    valPtr = b.create(luac.into_alloca, val=val, loc=op.loc).res()
    b.replace(op, [valPtr])
    return True

def convertLuaoptTableSetPrealloc(op, b):
    impl = b.create(luallvm.get_impl_direct, ref=op.tbl(), loc=op.loc).impl()
    arrPtr = b.create(LLVMBitcastOp, res=luallvm.ref(), arg=impl,
                      loc=op.loc).res()
    elPtr = b.create(LLVMGEPOp, res=luallvm.ref(), base=arrPtr,
                     indices=[op.iv()], loc=op.loc).res()
    ty, u = unpackTyAndU(b, op.val(), op.loc)
    b.create(luallvm.set_type_direct, ref=elPtr, type=ty, loc=op.loc)
    b.create(luallvm.set_u_direct, ref=elPtr, u=u, loc=op.loc)

    #impl = b.create(luallvm.get_impl_direct, ref=op.tbl(), loc=op.loc).impl()
    #val = loadRef(b, op.val(), op.loc)
    #b.create(luallvm.table_set_prealloc_impl, impl=impl, iv=op.iv(), val=val,
    #         loc=op.loc)
    b.erase(op)
    return True

def convertLuacMakeFcn(op, b):
    ref = allocaTyped(b, luac.type_fcn(), op.loc)
    impl = b.create(luallvm.make_fcn_impl, addr=op.addr(), capture=op.capture(),
                    loc=op.loc).impl()
    b.create(luallvm.set_impl_direct, ref=ref, impl=impl, loc=op.loc)
    b.replace(op, [ref])
    return True

def convertLuacGetImpl(op, b):
    impl = b.create(luallvm.get_impl_direct, ref=op.val(), loc=op.loc).impl()
    b.replace(op, [impl])
    return True

def convertLuacGetType(op, b):
    ty = b.create(luallvm.get_type_direct, ref=op.val(), loc=op.loc).type()
    b.replace(op, [ty])
    return True

def convertLuacGetBoolVal(op, b):
    u = b.create(luallvm.get_u_direct, ref=op.val(), loc=op.loc).u()
    # Truncate from i64 back to i1
    bVal = b.create(LLVMTruncOp, res=LLVMType.Int1(), value=u, loc=op.loc).res()
    b.replace(op, [bVal])
    return True

def convertLuacGetIntVal(op, b):
    u = b.create(luallvm.get_u_direct, ref=op.val(), loc=op.loc).u()
    b.replace(op, [u])
    return True

def convertLuacGetDoubleVal(op, b):
    u = b.create(luallvm.get_u_direct, ref=op.val(), loc=op.loc).u()
    # Bitcast i64 to double
    double = b.create(LLVMBitcastOp, res=LLVMType.Double(), arg=u,
                      loc=op.loc).res()
    b.replace(op, [double])
    return True

def convertLuaBuiltin(op, b):
    name = StringAttr("lua_builtin_" + op.var().getValue())
    builtin = b.create(luallvm.load_builtin, builtin=name, loc=op.loc).val()
    ref = b.create(luac.into_alloca, val=builtin, loc=op.loc).res()
    b.replace(op, [ref])
    return True

malloc = None
realloc = None

def convertLuacNewCapture(op, b):
    # Assumes 64-bit pointers, sizeof(TObject *) == 8
    ptrSz = b.create(ConstantOp, value=I32Attr(8), loc=op.loc).result()
    memSz = b.create(MulIOp, ty=I32Type(), lhs=ptrSz, rhs=op.size(),
                     loc=op.loc).result()
    memSzI64 = b.create(IndexCastOp, source=memSz, type=IndexType(),
                        loc=op.loc).result()
    ptr = b.create(CallOp, callee=malloc, operands=[memSzI64],
                   loc=op.loc).getResult(0)
    capture = b.create(LLVMBitcastOp, res=luallvm.capture(), arg=ptr,
                       loc=op.loc).res()
    b.replace(op, [capture])
    return True

def convertLuacAddCapture(op, b):
    elPtr = b.create(LLVMGEPOp, res=luallvm.capture(), base=op.capture(),
                     indices=[op.idx()], loc=op.loc).res()
    b.create(LLVMStoreOp, value=op.val(), addr=elPtr, loc=op.loc)
    b.erase(op)
    return True

def convertLuacGetCapture(op, b):
    elPtr = b.create(LLVMGEPOp, res=luallvm.capture(), base=op.capture(),
                     indices=[op.idx()], loc=op.loc).res()
    ref = b.create(LLVMLoadOp, res=luallvm.ref(), addr=elPtr, loc=op.loc).res()
    b.replace(op, [ref])
    return True

def luaToLLVMFirstPass(module):
    global malloc, realloc
    malloc = FuncOp("malloc", FunctionType([IndexType()], [LLVMType.Int8Ptr()]))
    realloc = FuncOp("realloc", FunctionType([LLVMType.Int8Ptr(), IndexType()],
                                             [LLVMType.Int8Ptr()]))
    module.append(malloc)
    module.append(realloc)
    applyOptPatterns(module, [
        Pattern(lua.nil, convertLuaNil),
        Pattern(lua.table, convertLuaTable),
        Pattern(luac.load_string, convertLuacLoadString),
        Pattern(luac.wrap_bool, convertLuacWrapBool),
        Pattern(luac.wrap_int, convertLuacWrapInt),
        Pattern(luac.wrap_real, convertLuacWrapReal),
        Pattern(lua.copy, convertLuaCopy),
        Pattern(lua.table_get, convertLuaTableGet),
        Pattern(lua.table_set, convertLuaTableSet),
        Pattern(luaopt.table_get_prealloc, convertLuaoptTableGetPrealloc),
        Pattern(luaopt.table_set_prealloc, convertLuaoptTableSetPrealloc),
        Pattern(luac.make_fcn, convertLuacMakeFcn),
        Pattern(luac.get_impl, convertLuacGetImpl),
        Pattern(luac.get_type, convertLuacGetType),
        Pattern(luac.get_bool_val, convertLuacGetBoolVal),
        Pattern(luac.get_int_val, convertLuacGetIntVal),
        Pattern(luac.get_double_val, convertLuacGetDoubleVal),
        Pattern(lua.builtin, convertLuaBuiltin),
        Pattern(luac.new_capture, convertLuacNewCapture),
        Pattern(luac.add_capture, convertLuacAddCapture),
        Pattern(luac.get_capture, convertLuacGetCapture),
    ])

################################################################################
# IR: Lua to LLVMIR Pass 2
################################################################################

def convertLuaLLVMAllocaValue(op, b):
    val = b.create(LLVMAllocaOp, res=luallvm.ref(),
                   arrSz=llvmI32Const(b, 1, op.loc), align=I64Attr(8),
                   loc=op.loc).res()
    b.replace(op, [val])
    return True

def convertLuaLLVMConstType(op, b):
    ty = b.create(LLVMConstantOp, res=luallvm.type(), value=op.value(),
                  loc=op.loc).res()
    b.replace(op, [ty])
    return True

def convertLuaLLVMGetTypeDirect(op, b):
    tyPtr = b.create(luallvm.get_type_ptr, ref=op.ref(), loc=op.loc).type_ptr()
    ty = b.create(luallvm.get_type, type_ptr=tyPtr, loc=op.loc).type()
    b.replace(op, [ty])
    return True

def convertLuaLLVMSetTypeDirect(op, b):
    tyPtr = b.create(luallvm.get_type_ptr, ref=op.ref(), loc=op.loc).type_ptr()
    b.create(luallvm.set_type, type_ptr=tyPtr, type=op.type(), loc=op.loc)
    b.erase(op)
    return True

def convertLuaLLVMGetUDirect(op, b):
    uPtr = b.create(luallvm.get_u_ptr, ref=op.ref(), loc=op.loc).u_ptr()
    u = b.create(luallvm.get_u, u_ptr=uPtr, loc=op.loc).u()
    b.replace(op, [u])
    return True

def convertLuaLLVMSetUDirect(op, b):
    uPtr = b.create(luallvm.get_u_ptr, ref=op.ref(), loc=op.loc).u_ptr()
    b.create(luallvm.set_u, u_ptr=uPtr, u=op.u(), loc=op.loc)
    b.erase(op)
    return True

def convertLuaLLVMGetImplDirect(op, b):
    uPtr = b.create(luallvm.get_u_ptr, ref=op.ref(), loc=op.loc).u_ptr()
    implPtr = b.create(luallvm.u_ptr_to_impl_ptr, u_ptr=uPtr,
                       loc=op.loc).impl_ptr()
    impl = b.create(luallvm.get_impl, impl_ptr=implPtr, loc=op.loc).impl()
    b.replace(op, [impl])
    return True

def convertLuaLLVMSetImplDirect(op, b):
    uPtr = b.create(luallvm.get_u_ptr, ref=op.ref(), loc=op.loc).u_ptr()
    implPtr = b.create(luallvm.u_ptr_to_impl_ptr, u_ptr=uPtr,
                       loc=op.loc).impl_ptr()
    b.create(luallvm.set_impl, impl_ptr=implPtr, impl=op.impl(), loc=op.loc)
    b.erase(op)
    return True

def convertLuaLLVMGetTypePtr(op, b):
    zero = llvmI32Const(b, 0, op.loc)
    tyPtr = b.create(LLVMGEPOp, res=luallvm.type_ptr(), base=op.ref(),
                     indices=[zero, zero], loc=op.loc).res()
    b.replace(op, [tyPtr])
    return True

def convertLuaLLVMGetType(op, b):
    ty = b.create(LLVMLoadOp, res=luallvm.type(), addr=op.type_ptr(),
                  loc=op.loc).res()
    b.replace(op, [ty])
    return True

def convertLuaLLVMSetType(op, b):
    b.create(LLVMStoreOp, value=op.type(), addr=op.type_ptr(), loc=op.loc)
    b.erase(op)
    return True

def convertLuaLLVMGetUPtr(op, b):
    zero = llvmI32Const(b, 0, op.loc)
    one = llvmI32Const(b, 1, op.loc)
    uPtr = b.create(LLVMGEPOp, res=luallvm.u_ptr(), base=op.ref(),
                    indices=[zero, one], loc=op.loc).res()
    b.replace(op, [uPtr])
    return True

def convertLuaLLVMGetU(op, b):
    u = b.create(LLVMLoadOp, res=luallvm.u(), addr=op.u_ptr(), loc=op.loc).res()
    b.replace(op, [u])
    return True

def convertLuaLLVMSetU(op, b):
    b.create(LLVMStoreOp, value=op.u(), addr=op.u_ptr(), loc=op.loc)
    b.erase(op)
    return True

def convertLuaLLVMUPtrToImplPtr(op, b):
    implPtr = b.create(LLVMBitcastOp, res=luallvm.impl_ptr(), arg=op.u_ptr(),
                       loc=op.loc).res()
    b.replace(op, [implPtr])
    return True

def convertLuaLLVMGetImpl(op, b):
    impl = b.create(LLVMLoadOp, res=luallvm.impl(), addr=op.impl_ptr(),
                    loc=op.loc).res()
    b.replace(op, [impl])
    return True

def convertLuaLLVMSetImpl(op, b):
    b.create(LLVMStoreOp, value=op.impl(), addr=op.impl_ptr(), loc=op.loc)
    b.erase(op)
    return True

def convertLuacGlobalString(op:luac.global_string, rewriter:Builder):
    i8Ty = LLVMType.Int8()
    i8ArrTy = LLVMType.ArrayOf(i8Ty, len(op.value().getValue()))
    rewriter.create(LLVMGlobalOp, ty=i8ArrTy, isConstant=False,
                    linkage=LLVMLinkage.Internal(), name=op.sym().getValue(),
                    value=op.value(), loc=op.loc)
    rewriter.erase(op)
    return True

def convertLuaLLVMGetStringData(module):
    def convert(op, b):
        symbol = module.lookup(op.sym().getValue())
        assert symbol, "cannot find global string " + str(op.sym())
        glob = LLVMGlobalOp(symbol)
        base = b.create(LLVMAddressOfOp, value=glob, loc=op.loc).res()
        zero = llvmI32Const(b, 0, op.loc)
        ptr = b.create(LLVMGEPOp, res=LLVMType.Int8Ptr(), base=base,
                       indices=[zero, zero], loc=op.loc).res()
        sz = llvmI64Const(b, len(glob.value().getValue()), op.loc)
        b.replace(op, [ptr, sz])
        return True
    return convert

# Why luac.into_alloca and luac.load_from must be broken down to element by
# element copy [i32, i64] instead of bulk copy { i32, i64 } to prevent strange
# errors is beyond me.
def convertLuacIntoAlloca(op, b):
    ref = b.create(luallvm.alloca_value, loc=op.loc).ref()
    ty = b.create(LLVMExtractValueOp, res=luallvm.type(), container=op.val(),
                  pos=I64ArrayAttr([0]), loc=op.loc).res()
    u = b.create(LLVMExtractValueOp, res=luallvm.u(), container=op.val(),
                 pos=I64ArrayAttr([1]), loc=op.loc).res()
    b.create(luallvm.set_type_direct, ref=ref, type=ty, loc=op.loc)
    b.create(luallvm.set_u_direct, ref=ref, u=u, loc=op.loc)
    b.replace(op, [ref])
    return True

def convertLuacLoadFrom(op, b):
    b.replace(op, [loadRef(b, op.val(), op.loc)])
    return True

def convertLuacPackInsert(op, b):
    objs = b.create(LLVMExtractValueOp, res=luallvm.ref(), container=op.pack(),
                    pos=I64ArrayAttr([1]), loc=op.loc).res()
    elPtr = b.create(LLVMGEPOp, res=luallvm.ref(), base=objs,
                     indices=[op.idx()], loc=op.loc).res()
    val = b.create(luac.load_from, val=op.val(), loc=op.loc).res()
    b.create(LLVMStoreOp, value=val, addr=elPtr, loc=op.loc)
    b.erase(op)
    return True

def convertLuacPackGetUnsafe(op, b):
    objs = b.create(LLVMExtractValueOp, res=luallvm.ref(), container=op.pack(),
                    pos=I64ArrayAttr([1]), loc=op.loc).res()
    elPtr = b.create(LLVMGEPOp, res=luallvm.ref(), base=objs,
                     indices=[op.idx()], loc=op.loc).res()
    val = b.create(LLVMLoadOp, res=luallvm.value(), addr=elPtr, loc=op.loc).res()
    ref = b.create(luac.into_alloca, val=val, loc=op.loc).res()
    b.replace(op, [ref])
    return True

def convertLuacPackGetSize(op, b):
    sz = b.create(LLVMExtractValueOp, res=LLVMType.Int32(), container=op.pack(),
                  pos=I64ArrayAttr([0]), loc=op.loc).res()
    b.replace(op, [sz])
    return True

def convertLuacGetFcnAddr(op, b):
    impl = b.create(luallvm.get_impl_direct, ref=op.val(), loc=op.loc).impl()
    cloPtr = b.create(LLVMBitcastOp, res=luallvm.closure_ptr(), arg=impl,
                      loc=op.loc).res()
    zero = llvmI32Const(b, 0, op.loc)
    addrPtr = b.create(LLVMGEPOp, res=luallvm.fcn_ptr(), base=cloPtr,
                       indices=[zero, zero], loc=op.loc).res()
    addr = b.create(LLVMLoadOp, res=luallvm.fcn(), addr=addrPtr,
                    loc=op.loc).res()
    b.replace(op, [addr])
    return True

def convertLuacGetCapturePack(op, b):
    impl = b.create(luallvm.get_impl_direct, ref=op.val(), loc=op.loc).impl()
    cloPtr = b.create(LLVMBitcastOp, res=luallvm.closure_ptr(), arg=impl,
                      loc=op.loc).res()
    zero = llvmI32Const(b, 0, op.loc)
    one = llvmI32Const(b, 1, op.loc)
    capPtr = b.create(LLVMGEPOp, res=luallvm.capture_ptr(), base=cloPtr,
                      indices=[zero, one], loc=op.loc).res()
    cap = b.create(LLVMLoadOp, res=luallvm.capture(), addr=capPtr,
                   loc=op.loc).res()
    b.replace(op, [cap])
    return True

def addBuiltins(module, names):
    for name in names:
        builtin = LLVMGlobalOp(luallvm.value(), False, LLVMLinkage.External(),
                               "lua_builtin_" + name, Attribute(), UnknownLoc())
        module.append(builtin)

def convertLuaLLVMLoadBuiltin(module):
    def convert(op, b):
        builtin = module.lookup(op.builtin().getValue())
        assert builtin, "failed to find built-in " + str(op.builtin())
        glob = LLVMGlobalOp(builtin)
        ptr = b.create(LLVMAddressOfOp, value=glob, loc=op.loc).res()
        val = b.create(LLVMLoadOp, res=luallvm.value(), addr=ptr,
                       loc=op.loc).res()
        b.replace(op, [val])
    return convert

def convertLuacGetPack(packPtr):
    def convert(op, b):
        base = b.create(LLVMAddressOfOp, value=packPtr, loc=op.loc).res()
        basePtr = b.create(LLVMBitcastOp, res=LLVMType.Int8Ptr().ptr_to(),
                           arg=base, loc=op.loc).res()
        ptr = b.create(LLVMLoadOp, res=LLVMType.Int8Ptr(), addr=basePtr,
                       loc=op.loc).res()
        undef = b.create(LLVMUndefOp, ty=luallvm.pack(), loc=op.loc).res()
        objs = b.create(LLVMBitcastOp, res=luallvm.ref(), arg=ptr,
                        loc=op.loc).res()
        v0 = b.create(LLVMInsertValueOp, res=luallvm.pack(), container=undef,
                      value=op.size(), pos=I64ArrayAttr([0]), loc=op.loc).res()
        v1 = b.create(LLVMInsertValueOp, res=luallvm.pack(), container=v0,
                      value=objs, pos=I64ArrayAttr([1]), loc=op.loc).res()
        b.replace(op, [v1])
    return convert

def prepMain(module, argPackPtr, retPackPtr):
    b = Builder()
    main = FuncOp("main", FunctionType([], [I32Type()]))
    b.insertAtStart(main.getBody().addEntryBlock([]))
    def giveMem(name, packPtr):
        mem = LLVMGlobalOp(LLVMType.ArrayOf(luallvm.value(), 16), False,
                           LLVMLinkage.Internal(), name, Attribute(),
                           UnknownLoc())
        module.append(mem)
        base = b.create(LLVMAddressOfOp, value=mem, loc=main.loc).res()
        zero = llvmI32Const(b, 0, main.loc)
        ptr = b.create(LLVMGEPOp, res=luallvm.ref(), base=base,
                       indices=[zero, zero], loc=main.loc).res()
        memPtr = b.create(LLVMPtrToIntOp, res=LLVMType.Int64(), arg=ptr,
                          loc=main.loc).res()
        tgt = b.create(LLVMAddressOfOp, value=packPtr, loc=main.loc).res()
        b.create(LLVMStoreOp, value=memPtr, addr=tgt, loc=main.loc)
    giveMem("g_arg_pack_mem", argPackPtr)
    giveMem("g_ret_pack_mem", retPackPtr)
    luaMain = module.lookup("lua_main")
    b.create(CallOp, callee=luaMain, operands=[], loc=main.loc)
    ok = b.create(ConstantOp, value=I32Attr(0), loc=main.loc).result()
    b.create(ReturnOp, operands=[ok], loc=main.loc)
    module.append(main)

def luaToLLVMSecondPass(module):
    argPackPtr = LLVMGlobalOp(LLVMType.Int64(), False, LLVMLinkage.Internal(),
                              "g_arg_pack_ptr", I64Attr(0), UnknownLoc())
    retPackPtr = LLVMGlobalOp(LLVMType.Int64(), False, LLVMLinkage.Internal(),
                              "g_ret_pack_ptr", I64Attr(0), UnknownLoc())
    module.append(argPackPtr)
    module.append(retPackPtr)
    prepMain(module, argPackPtr, retPackPtr)
    addBuiltins(module, list(lua_builtins))

    applyOptPatterns(module, [
        Pattern(luac.global_string, convertLuacGlobalString),
        Pattern(luallvm.get_string_data, convertLuaLLVMGetStringData(module)),

        Pattern(luac.get_arg_pack, convertLuacGetPack(argPackPtr)),
        Pattern(luac.get_ret_pack, convertLuacGetPack(retPackPtr)),
        Pattern(luac.pack_insert, convertLuacPackInsert),
        Pattern(luac.pack_get_unsafe, convertLuacPackGetUnsafe),
        Pattern(luac.pack_get_size, convertLuacPackGetSize),
        Pattern(luac.get_fcn_addr, convertLuacGetFcnAddr),
        Pattern(luac.get_capture_pack, convertLuacGetCapturePack),

        Pattern(luallvm.load_builtin, convertLuaLLVMLoadBuiltin(module)),
    ])
    luaToLLVMLatePass(module)

def luaToLLVMLatePass(module):
    applyOptPatterns(module, [
        Pattern(luallvm.alloca_value, convertLuaLLVMAllocaValue),
        Pattern(luallvm.const_type, convertLuaLLVMConstType),
        Pattern(luallvm.get_type_direct, convertLuaLLVMGetTypeDirect),
        Pattern(luallvm.set_type_direct, convertLuaLLVMSetTypeDirect),
        Pattern(luallvm.get_u_direct, convertLuaLLVMGetUDirect),
        Pattern(luallvm.set_u_direct, convertLuaLLVMSetUDirect),
        Pattern(luallvm.get_impl_direct, convertLuaLLVMGetImplDirect),
        Pattern(luallvm.set_impl_direct, convertLuaLLVMSetImplDirect),

        Pattern(luallvm.get_type_ptr, convertLuaLLVMGetTypePtr),
        Pattern(luallvm.get_type, convertLuaLLVMGetType),
        Pattern(luallvm.set_type, convertLuaLLVMSetType),
        Pattern(luallvm.get_u_ptr, convertLuaLLVMGetUPtr),
        Pattern(luallvm.get_u, convertLuaLLVMGetU),
        Pattern(luallvm.set_u, convertLuaLLVMSetU),
        Pattern(luallvm.u_ptr_to_impl_ptr, convertLuaLLVMUPtrToImplPtr),
        Pattern(luallvm.get_impl, convertLuaLLVMGetImpl),
        Pattern(luallvm.set_impl, convertLuaLLVMSetImpl),

        Pattern(luac.into_alloca, convertLuacIntoAlloca),
        Pattern(luac.load_from, convertLuacLoadFrom),
    ])

################################################################################
# IR: Lua to LLVMIR Pass 3
################################################################################

def convertToLibCall(module:ModuleOp, funcName:str, needWrap):
    def convertFcn(op:Operation, b:Builder):
        rawOp = module.lookup(funcName)
        assert rawOp, "cannot find lib.mlir function '" + funcName + "'"
        if needWrap:
            args = []
            for arg in op.getOperands():
                if arg.type == luallvm.ref() or arg.type == lua.val():
                    wrap = b.create(luac.load_from, val=arg, loc=op.loc).res()
                    args.append(wrap)
                else:
                    args.append(arg)
        else:
            args = op.getOperands()
        call = b.create(CallOp, callee=rawOp, operands=args, loc=op.loc)
        if needWrap:
            results = []
            for res in call.getResults():
                if res.type == luallvm.value() or res.type == lua.val():
                    wrap = b.create(luac.into_alloca, val=res, loc=op.loc).res()
                    results.append(wrap)
                else:
                    results.append(res)
        else:
            results = call.getResults()

        if len(results) == 0:
            b.erase(op)
        else:
            b.replace(op, results)
        return True
    return convertFcn

def luaToLLVMThirdPass(module):
    def convert(opCls, funcName:str):
        return Pattern(opCls, convertToLibCall(module, funcName, False))
    def convertWrap(opCls, funcName:str):
        return Pattern(opCls, convertToLibCall(module, funcName, True))

    applyOptPatterns(module, [
        convertWrap(luac.add, "lua_add"),
        convertWrap(luac.sub, "lua_sub"),
        convertWrap(luac.mul, "lua_mul"),
        convertWrap(luac.pow, "lua_pow"),
        convertWrap(luac.lt, "lua_lt"),
        convertWrap(luac.le, "lua_le"),
        convertWrap(luac.ge, "lua_ge"),
        convertWrap(luac.gt, "lua_gt"),
        convertWrap(luac.eq, "lua_eq"),
        convertWrap(luac.ne, "lua_ne"),
        convertWrap(luac.bool_and, "lua_bool_and"),
        convertWrap(luac.bool_not, "lua_bool_not"),
        convertWrap(luac.list_size, "lua_list_size"),
        convertWrap(luac.neg, "lua_neg"),
        convertWrap(luac.strcat, "lua_strcat"),
        convertWrap(luac.convert_bool_like, "lua_convert_bool_like"),
        convertWrap(luac.pack_insert_all, "lua_pack_insert_all"),
        convertWrap(luac.pack_get, "lua_pack_get"),

        convert(luallvm.table_get_impl, "lua_table_get_impl"),
        convert(luallvm.table_set_impl, "lua_table_set_impl"),
        convert(luallvm.table_get_prealloc_impl, "lua_table_get_prealloc_impl"),
        convert(luallvm.table_set_prealloc_impl, "lua_table_set_prealloc_impl"),
        convert(luallvm.make_fcn_impl, "lua_make_fcn_impl"),
        convert(luallvm.load_string_impl, "lua_load_string_impl"),
        convert(luallvm.new_table_impl, "lua_new_table_impl"),
    ])
    luaToLLVMLatePass(module)
    target = LLVMConversionTarget()
    lowerToLLVM(module, LLVMConversionTarget(), [], [
        lambda ty: luallvm.value() if ty == lua.val() else None,
        lambda ty: luallvm.pack() if ty == lua.pack() else None,
        lambda ty: luallvm.capture() if ty == lua.capture() else None,
        lambda ty: luallvm.impl() if ty == luac.void_ptr() else None,
    ])

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
    varAllocPass(module, main)
    cfExpand(module, main)
    applyOpts(module)

    lib = parseSourceFile("lib.mlir")
    for func in lib.getOps(FuncOp):
            module.append(func.clone())

    lowerToLuac(module)
    lowerSCFToStandard(module)
    luaToLLVMFirstPass(module)
    luaToLLVMSecondPass(module)
    luaToLLVMThirdPass(module)
    print(module)
    verify(module)

if __name__ == '__main__':
    main()
