module {
  func @test(%arg0 : !stencil.temp<[-1, -1, -1], f32>) -> f32 {
    %0 = "stencil.access"(%arg0) { offset = [0, 0, 0] } : (!stencil.temp<[-1, -1, -1], f32>) -> f32
    return %0 : f32
  }
}
