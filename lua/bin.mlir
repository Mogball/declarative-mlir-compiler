func @main() {
  %0 = lua.alloc
  %1 = lua.alloc
  %2 = lua.alloc
  %3 = lua.alloc
  %4 = lua.alloc
  %5 = lua.alloc
  %6 = lua.number 5 : i64
  %7 = lua.number 6 : i64
  %8 = lua.concat(%6, %7) : (!lua.ref, !lua.ref) -> !lua.pack
  %9 = lua.call %5(%8)
  %10 = lua.concat(%0, %1) : (!lua.ref, !lua.ref) -> !lua.pack
  %11 = lua.call %5(%10)
  %12 = lua.concat(%2, %3, %4) : (!lua.ref, !lua.ref, !lua.ref) -> !lua.pack
  %13 = lua.call %5(%12)
  %14 = lua.alloc
  %15 = lua.binary %1 "*" %2
  %16 = lua.binary %0 "+" %15
  %17 = lua.binary %0 "+" %1
  %18 = lua.binary %17 "*" %2
  %19 = lua.concat(%14, %16, %18) : (!lua.ref, !lua.ref, !lua.ref) -> !lua.pack
  %20 = lua.call %5(%19)
  %21 = lua.concat(%1) : (!lua.ref) -> !lua.pack
  %22 = lua.call %5(%21)
  %23 = lua.concat(%0, %22, %2) : (!lua.ref, !lua.pack, !lua.ref) -> !lua.pack
  %24 = lua.call %5(%23)
  %25 = lua.concat(%1) : (!lua.ref) -> !lua.pack
  %26 = lua.call %5(%25)
  return
}
