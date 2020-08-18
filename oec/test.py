import inspect, ast
import astpretty
import numpy

stencil = None

class StencilProgramVisitor(ast.NodeVisitor):
    def __init__(self):
        self.kinds = set()

    def generic_visit(self, node):
        self.kinds.add(node.__class__.__name__)
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_Module(self, node):
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        return self.generic_visit(node)

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
