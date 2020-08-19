import inspect, ast
import astpretty
import numpy

class StencilProgramVisitor(ast.NodeVisitor):
    def __init__(self):
        self.kinds = set()

    def generic_visit(self, node):
        self.kinds.add(node.__class__.__name__)
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_Module(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_FunctionDef(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_arguments(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_arg(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_Assign(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_Expr(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_Call(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_Return(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_Name(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_Load(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_Store(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_Attribute(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_Subscript(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_Index(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_Constant(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_Tuple(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_List(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_UnaryOp(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_USub(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_BinOp(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_Add(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_Mult(self, node):
        return ast.NodeVisitor.generic_visit(self, node)


stencil = None

def laplace(a:numpy.ndarray, b:numpy.ndarray):
  stencil.cast(a, [-4, -4, -4], [68, 68, 68])
  stencil.cast(b, [-4, -4, -4], [68, 68, 68])
  atmp = stencil.load(a)

  def applyFcn(c):
    return -4 * c[0, 0, 0] + c[-1, 0, 0] + c[1, 0, 0] + c[0, 1, 0] + c[0, -1, 0]

  btmp = stencil.apply(atmp, applyFcn)
  stencil.store(b, btmp, [0, 0, 0], [64, 64, 64])

node = ast.parse(inspect.getsource(laplace))
visitor = StencilProgramVisitor()
visitor.visit(node)
print(visitor.kinds)
astpretty.pprint(node)
