

module {
  func @laplace(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.apply (%arg2 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %2 = stencil.access %arg2 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %3 = stencil.access %arg2 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %4 = stencil.access %arg2 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %5 = stencil.access %arg2 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %7 = addf %2, %3 : f64
      %8 = addf %4, %5 : f64
      %9 = addf %7, %8 : f64
      %cst = constant -4.000000e+00 : f64
      %10 = mulf %6, %cst : f64
      %11 = addf %10, %9 : f64
      stencil.return %11 : f64
    }
    stencil.store %1 to %arg1([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }

  func @fill(%arg0: memref<72x72x72xf64>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c72 = constant 72 : index
    scf.parallel (%i, %j, %k) = (%c0, %c0, %c0) to (%c72, %c72, %c72) step (%c1, %c1, %c1) {
      %0 = index_cast %i : index to i64
      %1 = index_cast %j : index to i64
      %2 = index_cast %k : index to i64
      %3 = sitofp %0 : i64 to f64
      %4 = sitofp %1 : i64 to f64
      %5 = sitofp %2 : i64 to f64
      %6 = addf %3, %4 : f64
      %7 = addf %6, %5 : f64
      store %7, %arg0[%i, %j, %k] : memref<72x72x72xf64>
      scf.yield
    }
    return
  }
}
