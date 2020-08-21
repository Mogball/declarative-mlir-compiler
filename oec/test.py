import stencil
import numpy as np

@stencil.program
def laplace(a, b):
  stencil.cast(a, [-4, -4, -4], [68, 68, 68])
  stencil.cast(b, [-4, -4, -4], [68, 68, 68])
  atmp = stencil.load(a)

  def applyFcn(c) -> float:
    return c[0, 0, 0] + c[-1, 0, 0] + c[1, 0, 0] + c[0, 1, 0] + c[0, -1, 0]

  btmp = stencil.apply(atmp, applyFcn)
  stencil.store(b, btmp, [0, 0, 0], [64, 64, 64])
  return

a = np.empty([72, 72, 72], dtype='d')
b = np.empty([72, 72, 72], dtype='d')
a.fill(3)
laplace(a, b)
print(b)
print(b[32,32,32])
