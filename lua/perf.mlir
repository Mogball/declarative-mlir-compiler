

module {
  llvm.func @lua_main() -> !llvm<"{ i32, { i32, i64 }* }"> {
    %0 = llvm.mlir.addressof @lua_anon_fcn_1 : !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*">
    %1 = llvm.mlir.addressof @lua_anon_fcn_0 : !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*">
    %2 = llvm.mlir.constant(4.000000e+00 : f64) : !llvm.double
    %3 = llvm.mlir.constant(1.800000e+01 : f64) : !llvm.double
    %4 = llvm.mlir.constant(1.000000e+00 : f64) : !llvm.double
    %5 = llvm.mlir.constant(2.000000e+00 : f64) : !llvm.double
    %6 = llvm.mlir.constant(0.000000e+00 : f64) : !llvm.double
    %7 = llvm.mlir.constant(5 : i32) : !llvm.i32
    %8 = llvm.mlir.constant(4 : i32) : !llvm.i32
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %11 = llvm.mlir.constant(3 : i32) : !llvm.i32
    %12 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %13 = llvm.mlir.constant(8 : index) : !llvm.i64
    %14 = llvm.mlir.constant(2 : index) : !llvm.i64
    %15 = llvm.mlir.constant(5 : index) : !llvm.i64
    %16 = llvm.mlir.constant(1 : index) : !llvm.i64
    %17 = llvm.mlir.constant(0 : index) : !llvm.i64
    %18 = llvm.mlir.addressof @lua_builtin_print : !llvm<"{ i32, i64 }*">
    %19 = llvm.load %18 : !llvm<"{ i32, i64 }*">
    %20 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %21 = llvm.alloca %20 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %22 = llvm.extractvalue %19[0] : !llvm<"{ i32, i64 }">
    %23 = llvm.extractvalue %19[1] : !llvm<"{ i32, i64 }">
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.getelementptr %21[%24, %24] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %22, %25 : !llvm<"i32*">
    %26 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %27 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %28 = llvm.getelementptr %21[%26, %27] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %23, %28 : !llvm<"i64*">
    %29 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %30 = llvm.alloca %29 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %31 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %32 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %33 = llvm.getelementptr %30[%32, %32] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %31, %33 : !llvm<"i32*">
    %34 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %35 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %36 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %37 = llvm.getelementptr %30[%35, %36] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %34, %37 : !llvm<"i64*">
    %38 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %39 = llvm.alloca %38 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %40 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %41 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %42 = llvm.getelementptr %39[%41, %41] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %40, %42 : !llvm<"i32*">
    %43 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %44 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %45 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %46 = llvm.getelementptr %39[%44, %45] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %43, %46 : !llvm<"i64*">
    %47 = llvm.call @malloc(%13) : (!llvm.i64) -> !llvm<"i8*">
    %48 = llvm.bitcast %47 : !llvm<"i8*"> to !llvm<"{ i32, i64 }**">
    %49 = llvm.getelementptr %48[%12] : (!llvm<"{ i32, i64 }**">, !llvm.i32) -> !llvm<"{ i32, i64 }**">
    llvm.store %39, %49 : !llvm<"{ i32, i64 }**">
    %50 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %51 = llvm.alloca %50 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %52 = llvm.mlir.constant(5 : i32) : !llvm.i32
    %53 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %54 = llvm.getelementptr %51[%53, %53] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %52, %54 : !llvm<"i32*">
    %55 = llvm.call @lua_make_fcn_impl(%0, %48) : (!llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*">, !llvm<"{ i32, i64 }**">) -> !llvm<"i8*">
    %56 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %57 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %58 = llvm.getelementptr %51[%56, %57] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %59 = llvm.bitcast %58 : !llvm<"i64*"> to !llvm<"i8**">
    llvm.store %55, %59 : !llvm<"i8**">
    %60 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %61 = llvm.getelementptr %51[%60, %60] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %62 = llvm.load %61 : !llvm<"i32*">
    %63 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %64 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %65 = llvm.getelementptr %51[%63, %64] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %66 = llvm.load %65 : !llvm<"i64*">
    %67 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %68 = llvm.getelementptr %39[%67, %67] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %62, %68 : !llvm<"i32*">
    %69 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %70 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %71 = llvm.getelementptr %39[%69, %70] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %66, %71 : !llvm<"i64*">
    %72 = llvm.call @malloc(%13) : (!llvm.i64) -> !llvm<"i8*">
    %73 = llvm.bitcast %72 : !llvm<"i8*"> to !llvm<"{ i32, i64 }**">
    %74 = llvm.getelementptr %73[%12] : (!llvm<"{ i32, i64 }**">, !llvm.i32) -> !llvm<"{ i32, i64 }**">
    llvm.store %30, %74 : !llvm<"{ i32, i64 }**">
    %75 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %76 = llvm.alloca %75 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %77 = llvm.mlir.constant(5 : i32) : !llvm.i32
    %78 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %79 = llvm.getelementptr %76[%78, %78] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %77, %79 : !llvm<"i32*">
    %80 = llvm.call @lua_make_fcn_impl(%1, %73) : (!llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*">, !llvm<"{ i32, i64 }**">) -> !llvm<"i8*">
    %81 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %82 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %83 = llvm.getelementptr %76[%81, %82] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %84 = llvm.bitcast %83 : !llvm<"i64*"> to !llvm<"i8**">
    llvm.store %80, %84 : !llvm<"i8**">
    %85 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %86 = llvm.getelementptr %76[%85, %85] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %87 = llvm.load %86 : !llvm<"i32*">
    %88 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %89 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %90 = llvm.getelementptr %76[%88, %89] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %91 = llvm.load %90 : !llvm<"i64*">
    %92 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %93 = llvm.getelementptr %30[%92, %92] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %87, %93 : !llvm<"i32*">
    %94 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %95 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %96 = llvm.getelementptr %30[%94, %95] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %91, %96 : !llvm<"i64*">
    %97 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %98 = llvm.alloca %97 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %99 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %100 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %101 = llvm.getelementptr %98[%100, %100] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %99, %101 : !llvm<"i32*">
    %102 = llvm.bitcast %2 : !llvm.double to !llvm.i64
    %103 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %104 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %105 = llvm.getelementptr %98[%103, %104] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %102, %105 : !llvm<"i64*">
    %106 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %107 = llvm.alloca %106 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %108 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %109 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %110 = llvm.getelementptr %107[%109, %109] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %108, %110 : !llvm<"i32*">
    %111 = llvm.bitcast %3 : !llvm.double to !llvm.i64
    %112 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %113 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %114 = llvm.getelementptr %107[%112, %113] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %111, %114 : !llvm<"i64*">
    %115 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %116 = llvm.alloca %115 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %117 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %118 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %119 = llvm.getelementptr %116[%118, %118] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %117, %119 : !llvm<"i32*">
    %120 = llvm.bitcast %4 : !llvm.double to !llvm.i64
    %121 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %122 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %123 = llvm.getelementptr %116[%121, %122] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %120, %123 : !llvm<"i64*">
    %124 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %125 = llvm.getelementptr %107[%124, %124] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %126 = llvm.load %125 : !llvm<"i32*">
    %127 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %128 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %129 = llvm.getelementptr %107[%127, %128] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %130 = llvm.load %129 : !llvm<"i64*">
    %131 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %132 = llvm.insertvalue %126, %131[0] : !llvm<"{ i32, i64 }">
    %133 = llvm.insertvalue %130, %132[1] : !llvm<"{ i32, i64 }">
    %134 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %135 = llvm.getelementptr %116[%134, %134] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %136 = llvm.load %135 : !llvm<"i32*">
    %137 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %138 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %139 = llvm.getelementptr %116[%137, %138] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %140 = llvm.load %139 : !llvm<"i64*">
    %141 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %142 = llvm.insertvalue %136, %141[0] : !llvm<"{ i32, i64 }">
    %143 = llvm.insertvalue %140, %142[1] : !llvm<"{ i32, i64 }">
    %144 = llvm.call @lua_add(%133, %143) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }">
    %145 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %146 = llvm.alloca %145 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %147 = llvm.extractvalue %144[0] : !llvm<"{ i32, i64 }">
    %148 = llvm.extractvalue %144[1] : !llvm<"{ i32, i64 }">
    %149 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %150 = llvm.getelementptr %146[%149, %149] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %147, %150 : !llvm<"i32*">
    %151 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %152 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %153 = llvm.getelementptr %146[%151, %152] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %148, %153 : !llvm<"i64*">
    %154 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %155 = llvm.alloca %154 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %156 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %157 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %158 = llvm.getelementptr %155[%157, %157] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %156, %158 : !llvm<"i32*">
    %159 = llvm.bitcast %6 : !llvm.double to !llvm.i64
    %160 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %161 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %162 = llvm.getelementptr %155[%160, %161] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %159, %162 : !llvm<"i64*">
    %163 = llvm.mlir.addressof @g_arg_pack_ptr : !llvm<"i64*">
    %164 = llvm.bitcast %163 : !llvm<"i64*"> to !llvm<"i8**">
    %165 = llvm.load %164 : !llvm<"i8**">
    %166 = llvm.call @realloc(%165, %14) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    llvm.store %166, %164 : !llvm<"i8**">
    %167 = llvm.mlir.undef : !llvm<"{ i32, { i32, i64 }* }">
    %168 = llvm.bitcast %166 : !llvm<"i8*"> to !llvm<"{ i32, i64 }*">
    %169 = llvm.insertvalue %10, %167[0] : !llvm<"{ i32, { i32, i64 }* }">
    %170 = llvm.insertvalue %168, %169[1] : !llvm<"{ i32, { i32, i64 }* }">
    %171 = llvm.extractvalue %170[1] : !llvm<"{ i32, { i32, i64 }* }">
    %172 = llvm.getelementptr %171[%12] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %173 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %174 = llvm.getelementptr %155[%173, %173] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %175 = llvm.load %174 : !llvm<"i32*">
    %176 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %177 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %178 = llvm.getelementptr %155[%176, %177] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %179 = llvm.load %178 : !llvm<"i64*">
    %180 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %181 = llvm.insertvalue %175, %180[0] : !llvm<"{ i32, i64 }">
    %182 = llvm.insertvalue %179, %181[1] : !llvm<"{ i32, i64 }">
    llvm.store %182, %172 : !llvm<"{ i32, i64 }*">
    %183 = llvm.extractvalue %170[1] : !llvm<"{ i32, { i32, i64 }* }">
    %184 = llvm.getelementptr %183[%9] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %185 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %186 = llvm.getelementptr %146[%185, %185] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %187 = llvm.load %186 : !llvm<"i32*">
    %188 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %189 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %190 = llvm.getelementptr %146[%188, %189] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %191 = llvm.load %190 : !llvm<"i64*">
    %192 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %193 = llvm.insertvalue %187, %192[0] : !llvm<"{ i32, i64 }">
    %194 = llvm.insertvalue %191, %193[1] : !llvm<"{ i32, i64 }">
    llvm.store %194, %184 : !llvm<"{ i32, i64 }*">
    %195 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %196 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %197 = llvm.getelementptr %39[%195, %196] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %198 = llvm.bitcast %197 : !llvm<"i64*"> to !llvm<"i8**">
    %199 = llvm.load %198 : !llvm<"i8**">
    %200 = llvm.bitcast %199 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %201 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %202 = llvm.getelementptr %200[%201, %201] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %203 = llvm.load %202 : !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %204 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %205 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %206 = llvm.getelementptr %39[%204, %205] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %207 = llvm.bitcast %206 : !llvm<"i64*"> to !llvm<"i8**">
    %208 = llvm.load %207 : !llvm<"i8**">
    %209 = llvm.bitcast %208 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %210 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %211 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %212 = llvm.getelementptr %209[%210, %211] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, i64 }***">
    %213 = llvm.load %212 : !llvm<"{ i32, i64 }***">
    %214 = llvm.call %203(%213, %170) : (!llvm<"{ i32, i64 }**">, !llvm<"{ i32, { i32, i64 }* }">) -> !llvm<"{ i32, { i32, i64 }* }">
    %215 = llvm.extractvalue %214[1] : !llvm<"{ i32, { i32, i64 }* }">
    %216 = llvm.getelementptr %215[%12] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %217 = llvm.load %216 : !llvm<"{ i32, i64 }*">
    %218 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %219 = llvm.alloca %218 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %220 = llvm.extractvalue %217[0] : !llvm<"{ i32, i64 }">
    %221 = llvm.extractvalue %217[1] : !llvm<"{ i32, i64 }">
    %222 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %223 = llvm.getelementptr %219[%222, %222] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %220, %223 : !llvm<"i32*">
    %224 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %225 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %226 = llvm.getelementptr %219[%224, %225] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %221, %226 : !llvm<"i64*">
    %227 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %228 = llvm.alloca %227 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %229 = llvm.mlir.constant(3 : i32) : !llvm.i32
    %230 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %231 = llvm.getelementptr %228[%230, %230] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %229, %231 : !llvm<"i32*">
    %232 = llvm.mlir.addressof @lua_anon_string_0 : !llvm<"[21 x i8]*">
    %233 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %234 = llvm.getelementptr %232[%233, %233] : (!llvm<"[21 x i8]*">, !llvm.i32, !llvm.i32) -> !llvm<"i8*">
    %235 = llvm.mlir.constant(21 : i64) : !llvm.i64
    %236 = llvm.call @lua_load_string_impl(%234, %235) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    %237 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %238 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %239 = llvm.getelementptr %228[%237, %238] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %240 = llvm.bitcast %239 : !llvm<"i64*"> to !llvm<"i8**">
    llvm.store %236, %240 : !llvm<"i8**">
    %241 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %242 = llvm.alloca %241 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %243 = llvm.mlir.constant(3 : i32) : !llvm.i32
    %244 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %245 = llvm.getelementptr %242[%244, %244] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %243, %245 : !llvm<"i32*">
    %246 = llvm.mlir.addressof @lua_anon_string_1 : !llvm<"[6 x i8]*">
    %247 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %248 = llvm.getelementptr %246[%247, %247] : (!llvm<"[6 x i8]*">, !llvm.i32, !llvm.i32) -> !llvm<"i8*">
    %249 = llvm.mlir.constant(6 : i64) : !llvm.i64
    %250 = llvm.call @lua_load_string_impl(%248, %249) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    %251 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %252 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %253 = llvm.getelementptr %242[%251, %252] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %254 = llvm.bitcast %253 : !llvm<"i64*"> to !llvm<"i8**">
    llvm.store %250, %254 : !llvm<"i8**">
    %255 = llvm.mlir.addressof @g_arg_pack_ptr : !llvm<"i64*">
    %256 = llvm.bitcast %255 : !llvm<"i64*"> to !llvm<"i8**">
    %257 = llvm.load %256 : !llvm<"i8**">
    %258 = llvm.call @realloc(%257, %16) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    llvm.store %258, %256 : !llvm<"i8**">
    %259 = llvm.mlir.undef : !llvm<"{ i32, { i32, i64 }* }">
    %260 = llvm.bitcast %258 : !llvm<"i8*"> to !llvm<"{ i32, i64 }*">
    %261 = llvm.insertvalue %9, %259[0] : !llvm<"{ i32, { i32, i64 }* }">
    %262 = llvm.insertvalue %260, %261[1] : !llvm<"{ i32, { i32, i64 }* }">
    %263 = llvm.extractvalue %262[1] : !llvm<"{ i32, { i32, i64 }* }">
    %264 = llvm.getelementptr %263[%12] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %265 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %266 = llvm.getelementptr %219[%265, %265] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %267 = llvm.load %266 : !llvm<"i32*">
    %268 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %269 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %270 = llvm.getelementptr %219[%268, %269] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %271 = llvm.load %270 : !llvm<"i64*">
    %272 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %273 = llvm.insertvalue %267, %272[0] : !llvm<"{ i32, i64 }">
    %274 = llvm.insertvalue %271, %273[1] : !llvm<"{ i32, i64 }">
    llvm.store %274, %264 : !llvm<"{ i32, i64 }*">
    %275 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %276 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %277 = llvm.getelementptr %30[%275, %276] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %278 = llvm.bitcast %277 : !llvm<"i64*"> to !llvm<"i8**">
    %279 = llvm.load %278 : !llvm<"i8**">
    %280 = llvm.bitcast %279 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %281 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %282 = llvm.getelementptr %280[%281, %281] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %283 = llvm.load %282 : !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %284 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %285 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %286 = llvm.getelementptr %30[%284, %285] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %287 = llvm.bitcast %286 : !llvm<"i64*"> to !llvm<"i8**">
    %288 = llvm.load %287 : !llvm<"i8**">
    %289 = llvm.bitcast %288 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %290 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %291 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %292 = llvm.getelementptr %289[%290, %291] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, i64 }***">
    %293 = llvm.load %292 : !llvm<"{ i32, i64 }***">
    %294 = llvm.call %283(%293, %262) : (!llvm<"{ i32, i64 }**">, !llvm<"{ i32, { i32, i64 }* }">) -> !llvm<"{ i32, { i32, i64 }* }">
    %295 = llvm.extractvalue %294[0] : !llvm<"{ i32, { i32, i64 }* }">
    %296 = llvm.add %295, %11 : !llvm.i32
    %297 = llvm.mlir.addressof @g_arg_pack_ptr : !llvm<"i64*">
    %298 = llvm.bitcast %297 : !llvm<"i64*"> to !llvm<"i8**">
    %299 = llvm.load %298 : !llvm<"i8**">
    %300 = llvm.sext %296 : !llvm.i32 to !llvm.i64
    %301 = llvm.call @realloc(%299, %300) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    llvm.store %301, %298 : !llvm<"i8**">
    %302 = llvm.mlir.undef : !llvm<"{ i32, { i32, i64 }* }">
    %303 = llvm.bitcast %301 : !llvm<"i8*"> to !llvm<"{ i32, i64 }*">
    %304 = llvm.insertvalue %296, %302[0] : !llvm<"{ i32, { i32, i64 }* }">
    %305 = llvm.insertvalue %303, %304[1] : !llvm<"{ i32, { i32, i64 }* }">
    %306 = llvm.extractvalue %305[1] : !llvm<"{ i32, { i32, i64 }* }">
    %307 = llvm.getelementptr %306[%12] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %308 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %309 = llvm.getelementptr %228[%308, %308] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %310 = llvm.load %309 : !llvm<"i32*">
    %311 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %312 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %313 = llvm.getelementptr %228[%311, %312] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %314 = llvm.load %313 : !llvm<"i64*">
    %315 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %316 = llvm.insertvalue %310, %315[0] : !llvm<"{ i32, i64 }">
    %317 = llvm.insertvalue %314, %316[1] : !llvm<"{ i32, i64 }">
    llvm.store %317, %307 : !llvm<"{ i32, i64 }*">
    %318 = llvm.extractvalue %305[1] : !llvm<"{ i32, { i32, i64 }* }">
    %319 = llvm.getelementptr %318[%9] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %320 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %321 = llvm.getelementptr %146[%320, %320] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %322 = llvm.load %321 : !llvm<"i32*">
    %323 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %324 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %325 = llvm.getelementptr %146[%323, %324] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %326 = llvm.load %325 : !llvm<"i64*">
    %327 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %328 = llvm.insertvalue %322, %327[0] : !llvm<"{ i32, i64 }">
    %329 = llvm.insertvalue %326, %328[1] : !llvm<"{ i32, i64 }">
    llvm.store %329, %319 : !llvm<"{ i32, i64 }*">
    %330 = llvm.extractvalue %305[1] : !llvm<"{ i32, { i32, i64 }* }">
    %331 = llvm.getelementptr %330[%10] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %332 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %333 = llvm.getelementptr %242[%332, %332] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %334 = llvm.load %333 : !llvm<"i32*">
    %335 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %336 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %337 = llvm.getelementptr %242[%335, %336] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %338 = llvm.load %337 : !llvm<"i64*">
    %339 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %340 = llvm.insertvalue %334, %339[0] : !llvm<"{ i32, i64 }">
    %341 = llvm.insertvalue %338, %340[1] : !llvm<"{ i32, i64 }">
    llvm.store %341, %331 : !llvm<"{ i32, i64 }*">
    llvm.call @lua_pack_insert_all(%305, %294, %11) : (!llvm<"{ i32, { i32, i64 }* }">, !llvm<"{ i32, { i32, i64 }* }">, !llvm.i32) -> ()
    %342 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %343 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %344 = llvm.getelementptr %21[%342, %343] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %345 = llvm.bitcast %344 : !llvm<"i64*"> to !llvm<"i8**">
    %346 = llvm.load %345 : !llvm<"i8**">
    %347 = llvm.bitcast %346 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %348 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %349 = llvm.getelementptr %347[%348, %348] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %350 = llvm.load %349 : !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %351 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %352 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %353 = llvm.getelementptr %21[%351, %352] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %354 = llvm.bitcast %353 : !llvm<"i64*"> to !llvm<"i8**">
    %355 = llvm.load %354 : !llvm<"i8**">
    %356 = llvm.bitcast %355 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %357 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %358 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %359 = llvm.getelementptr %356[%357, %358] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, i64 }***">
    %360 = llvm.load %359 : !llvm<"{ i32, i64 }***">
    %361 = llvm.call %350(%360, %305) : (!llvm<"{ i32, i64 }**">, !llvm<"{ i32, { i32, i64 }* }">) -> !llvm<"{ i32, { i32, i64 }* }">
    %362 = llvm.mlir.addressof @g_arg_pack_ptr : !llvm<"i64*">
    %363 = llvm.bitcast %362 : !llvm<"i64*"> to !llvm<"i8**">
    %364 = llvm.load %363 : !llvm<"i8**">
    %365 = llvm.call @realloc(%364, %14) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    llvm.store %365, %363 : !llvm<"i8**">
    %366 = llvm.mlir.undef : !llvm<"{ i32, { i32, i64 }* }">
    %367 = llvm.bitcast %365 : !llvm<"i8*"> to !llvm<"{ i32, i64 }*">
    %368 = llvm.insertvalue %10, %366[0] : !llvm<"{ i32, { i32, i64 }* }">
    %369 = llvm.insertvalue %367, %368[1] : !llvm<"{ i32, { i32, i64 }* }">
    %370 = llvm.extractvalue %369[1] : !llvm<"{ i32, { i32, i64 }* }">
    %371 = llvm.getelementptr %370[%12] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %372 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %373 = llvm.getelementptr %155[%372, %372] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %374 = llvm.load %373 : !llvm<"i32*">
    %375 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %376 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %377 = llvm.getelementptr %155[%375, %376] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %378 = llvm.load %377 : !llvm<"i64*">
    %379 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %380 = llvm.insertvalue %374, %379[0] : !llvm<"{ i32, i64 }">
    %381 = llvm.insertvalue %378, %380[1] : !llvm<"{ i32, i64 }">
    llvm.store %381, %371 : !llvm<"{ i32, i64 }*">
    %382 = llvm.extractvalue %369[1] : !llvm<"{ i32, { i32, i64 }* }">
    %383 = llvm.getelementptr %382[%9] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %384 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %385 = llvm.getelementptr %107[%384, %384] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %386 = llvm.load %385 : !llvm<"i32*">
    %387 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %388 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %389 = llvm.getelementptr %107[%387, %388] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %390 = llvm.load %389 : !llvm<"i64*">
    %391 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %392 = llvm.insertvalue %386, %391[0] : !llvm<"{ i32, i64 }">
    %393 = llvm.insertvalue %390, %392[1] : !llvm<"{ i32, i64 }">
    llvm.store %393, %383 : !llvm<"{ i32, i64 }*">
    %394 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %395 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %396 = llvm.getelementptr %39[%394, %395] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %397 = llvm.bitcast %396 : !llvm<"i64*"> to !llvm<"i8**">
    %398 = llvm.load %397 : !llvm<"i8**">
    %399 = llvm.bitcast %398 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %400 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %401 = llvm.getelementptr %399[%400, %400] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %402 = llvm.load %401 : !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %403 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %404 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %405 = llvm.getelementptr %39[%403, %404] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %406 = llvm.bitcast %405 : !llvm<"i64*"> to !llvm<"i8**">
    %407 = llvm.load %406 : !llvm<"i8**">
    %408 = llvm.bitcast %407 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %409 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %410 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %411 = llvm.getelementptr %408[%409, %410] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, i64 }***">
    %412 = llvm.load %411 : !llvm<"{ i32, i64 }***">
    %413 = llvm.call %402(%412, %369) : (!llvm<"{ i32, i64 }**">, !llvm<"{ i32, { i32, i64 }* }">) -> !llvm<"{ i32, { i32, i64 }* }">
    %414 = llvm.extractvalue %413[1] : !llvm<"{ i32, { i32, i64 }* }">
    %415 = llvm.getelementptr %414[%12] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %416 = llvm.load %415 : !llvm<"{ i32, i64 }*">
    %417 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %418 = llvm.alloca %417 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %419 = llvm.extractvalue %416[0] : !llvm<"{ i32, i64 }">
    %420 = llvm.extractvalue %416[1] : !llvm<"{ i32, i64 }">
    %421 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %422 = llvm.getelementptr %418[%421, %421] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %419, %422 : !llvm<"i32*">
    %423 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %424 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %425 = llvm.getelementptr %418[%423, %424] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %420, %425 : !llvm<"i64*">
    %426 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %427 = llvm.alloca %426 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %428 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %429 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %430 = llvm.getelementptr %427[%429, %429] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %428, %430 : !llvm<"i32*">
    %431 = llvm.bitcast %5 : !llvm.double to !llvm.i64
    %432 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %433 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %434 = llvm.getelementptr %427[%432, %433] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %431, %434 : !llvm<"i64*">
    %435 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %436 = llvm.getelementptr %116[%435, %435] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %437 = llvm.load %436 : !llvm<"i32*">
    %438 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %439 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %440 = llvm.getelementptr %116[%438, %439] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %441 = llvm.load %440 : !llvm<"i64*">
    %442 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %443 = llvm.insertvalue %437, %442[0] : !llvm<"{ i32, i64 }">
    %444 = llvm.insertvalue %441, %443[1] : !llvm<"{ i32, i64 }">
    %445 = llvm.call @lua_neg(%444) : (!llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }">
    %446 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %447 = llvm.alloca %446 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %448 = llvm.extractvalue %445[0] : !llvm<"{ i32, i64 }">
    %449 = llvm.extractvalue %445[1] : !llvm<"{ i32, i64 }">
    %450 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %451 = llvm.getelementptr %447[%450, %450] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %448, %451 : !llvm<"i32*">
    %452 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %453 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %454 = llvm.getelementptr %447[%452, %453] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %449, %454 : !llvm<"i64*">
    %455 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %456 = llvm.alloca %455 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %457 = llvm.mlir.constant(3 : i32) : !llvm.i32
    %458 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %459 = llvm.getelementptr %456[%458, %458] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %457, %459 : !llvm<"i32*">
    %460 = llvm.mlir.addressof @lua_anon_string_2 : !llvm<"[14 x i8]*">
    %461 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %462 = llvm.getelementptr %460[%461, %461] : (!llvm<"[14 x i8]*">, !llvm.i32, !llvm.i32) -> !llvm<"i8*">
    %463 = llvm.mlir.constant(14 : i64) : !llvm.i64
    %464 = llvm.call @lua_load_string_impl(%462, %463) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    %465 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %466 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %467 = llvm.getelementptr %456[%465, %466] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %468 = llvm.bitcast %467 : !llvm<"i64*"> to !llvm<"i8**">
    llvm.store %464, %468 : !llvm<"i8**">
    %469 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %470 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %471 = llvm.getelementptr %427[%469, %470] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %472 = llvm.load %471 : !llvm<"i64*">
    %473 = llvm.bitcast %472 : !llvm.i64 to !llvm.double
    %474 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %475 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %476 = llvm.getelementptr %98[%474, %475] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %477 = llvm.load %476 : !llvm<"i64*">
    %478 = llvm.bitcast %477 : !llvm.i64 to !llvm.double
    %479 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %480 = llvm.alloca %479 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %481 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %482 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %483 = llvm.getelementptr %480[%482, %482] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %481, %483 : !llvm<"i32*">
    %484 = llvm.bitcast %478 : !llvm.double to !llvm.i64
    %485 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %486 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %487 = llvm.getelementptr %480[%485, %486] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %484, %487 : !llvm<"i64*">
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb5
    %488 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %489 = llvm.getelementptr %480[%488, %488] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %490 = llvm.load %489 : !llvm<"i32*">
    %491 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %492 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %493 = llvm.getelementptr %480[%491, %492] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %494 = llvm.load %493 : !llvm<"i64*">
    %495 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %496 = llvm.insertvalue %490, %495[0] : !llvm<"{ i32, i64 }">
    %497 = llvm.insertvalue %494, %496[1] : !llvm<"{ i32, i64 }">
    %498 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %499 = llvm.getelementptr %107[%498, %498] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %500 = llvm.load %499 : !llvm<"i32*">
    %501 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %502 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %503 = llvm.getelementptr %107[%501, %502] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %504 = llvm.load %503 : !llvm<"i64*">
    %505 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %506 = llvm.insertvalue %500, %505[0] : !llvm<"{ i32, i64 }">
    %507 = llvm.insertvalue %504, %506[1] : !llvm<"{ i32, i64 }">
    %508 = llvm.call @lua_le(%497, %507) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }">
    %509 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %510 = llvm.alloca %509 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %511 = llvm.extractvalue %508[0] : !llvm<"{ i32, i64 }">
    %512 = llvm.extractvalue %508[1] : !llvm<"{ i32, i64 }">
    %513 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %514 = llvm.getelementptr %510[%513, %513] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %511, %514 : !llvm<"i32*">
    %515 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %516 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %517 = llvm.getelementptr %510[%515, %516] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %512, %517 : !llvm<"i64*">
    %518 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %519 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %520 = llvm.getelementptr %510[%518, %519] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %521 = llvm.load %520 : !llvm<"i64*">
    %522 = llvm.trunc %521 : !llvm.i64 to !llvm.i1
    llvm.cond_br %522, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %523 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %524 = llvm.getelementptr %107[%523, %523] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %525 = llvm.load %524 : !llvm<"i32*">
    %526 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %527 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %528 = llvm.getelementptr %107[%526, %527] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %529 = llvm.load %528 : !llvm<"i64*">
    %530 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %531 = llvm.insertvalue %525, %530[0] : !llvm<"{ i32, i64 }">
    %532 = llvm.insertvalue %529, %531[1] : !llvm<"{ i32, i64 }">
    %533 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %534 = llvm.getelementptr %480[%533, %533] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %535 = llvm.load %534 : !llvm<"i32*">
    %536 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %537 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %538 = llvm.getelementptr %480[%536, %537] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %539 = llvm.load %538 : !llvm<"i64*">
    %540 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %541 = llvm.insertvalue %535, %540[0] : !llvm<"{ i32, i64 }">
    %542 = llvm.insertvalue %539, %541[1] : !llvm<"{ i32, i64 }">
    %543 = llvm.call @lua_sub(%532, %542) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }">
    %544 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %545 = llvm.alloca %544 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %546 = llvm.extractvalue %543[0] : !llvm<"{ i32, i64 }">
    %547 = llvm.extractvalue %543[1] : !llvm<"{ i32, i64 }">
    %548 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %549 = llvm.getelementptr %545[%548, %548] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %546, %549 : !llvm<"i32*">
    %550 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %551 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %552 = llvm.getelementptr %545[%550, %551] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %547, %552 : !llvm<"i64*">
    %553 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %554 = llvm.getelementptr %545[%553, %553] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %555 = llvm.load %554 : !llvm<"i32*">
    %556 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %557 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %558 = llvm.getelementptr %545[%556, %557] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %559 = llvm.load %558 : !llvm<"i64*">
    %560 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %561 = llvm.insertvalue %555, %560[0] : !llvm<"{ i32, i64 }">
    %562 = llvm.insertvalue %559, %561[1] : !llvm<"{ i32, i64 }">
    %563 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %564 = llvm.getelementptr %98[%563, %563] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %565 = llvm.load %564 : !llvm<"i32*">
    %566 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %567 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %568 = llvm.getelementptr %98[%566, %567] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %569 = llvm.load %568 : !llvm<"i64*">
    %570 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %571 = llvm.insertvalue %565, %570[0] : !llvm<"{ i32, i64 }">
    %572 = llvm.insertvalue %569, %571[1] : !llvm<"{ i32, i64 }">
    %573 = llvm.call @lua_add(%562, %572) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }">
    %574 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %575 = llvm.alloca %574 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %576 = llvm.extractvalue %573[0] : !llvm<"{ i32, i64 }">
    %577 = llvm.extractvalue %573[1] : !llvm<"{ i32, i64 }">
    %578 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %579 = llvm.getelementptr %575[%578, %578] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %576, %579 : !llvm<"i32*">
    %580 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %581 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %582 = llvm.getelementptr %575[%580, %581] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %577, %582 : !llvm<"i64*">
    %583 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %584 = llvm.getelementptr %427[%583, %583] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %585 = llvm.load %584 : !llvm<"i32*">
    %586 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %587 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %588 = llvm.getelementptr %427[%586, %587] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %589 = llvm.load %588 : !llvm<"i64*">
    %590 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %591 = llvm.insertvalue %585, %590[0] : !llvm<"{ i32, i64 }">
    %592 = llvm.insertvalue %589, %591[1] : !llvm<"{ i32, i64 }">
    %593 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %594 = llvm.getelementptr %575[%593, %593] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %595 = llvm.load %594 : !llvm<"i32*">
    %596 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %597 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %598 = llvm.getelementptr %575[%596, %597] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %599 = llvm.load %598 : !llvm<"i64*">
    %600 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %601 = llvm.insertvalue %595, %600[0] : !llvm<"{ i32, i64 }">
    %602 = llvm.insertvalue %599, %601[1] : !llvm<"{ i32, i64 }">
    %603 = llvm.call @lua_pow(%592, %602) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }">
    %604 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %605 = llvm.alloca %604 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %606 = llvm.extractvalue %603[0] : !llvm<"{ i32, i64 }">
    %607 = llvm.extractvalue %603[1] : !llvm<"{ i32, i64 }">
    %608 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %609 = llvm.getelementptr %605[%608, %608] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %606, %609 : !llvm<"i32*">
    %610 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %611 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %612 = llvm.getelementptr %605[%610, %611] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %607, %612 : !llvm<"i64*">
    %613 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %614 = llvm.alloca %613 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %615 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %616 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %617 = llvm.getelementptr %614[%616, %616] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %615, %617 : !llvm<"i32*">
    %618 = llvm.bitcast %6 : !llvm.double to !llvm.i64
    %619 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %620 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %621 = llvm.getelementptr %614[%619, %620] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %618, %621 : !llvm<"i64*">
    %622 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %623 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %624 = llvm.getelementptr %116[%622, %623] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %625 = llvm.load %624 : !llvm<"i64*">
    %626 = llvm.bitcast %625 : !llvm.i64 to !llvm.double
    %627 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %628 = llvm.alloca %627 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %629 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %630 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %631 = llvm.getelementptr %628[%630, %630] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %629, %631 : !llvm<"i32*">
    %632 = llvm.bitcast %626 : !llvm.double to !llvm.i64
    %633 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %634 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %635 = llvm.getelementptr %628[%633, %634] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %632, %635 : !llvm<"i64*">
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb2, ^bb4
    %636 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %637 = llvm.getelementptr %628[%636, %636] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %638 = llvm.load %637 : !llvm<"i32*">
    %639 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %640 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %641 = llvm.getelementptr %628[%639, %640] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %642 = llvm.load %641 : !llvm<"i64*">
    %643 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %644 = llvm.insertvalue %638, %643[0] : !llvm<"{ i32, i64 }">
    %645 = llvm.insertvalue %642, %644[1] : !llvm<"{ i32, i64 }">
    %646 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %647 = llvm.getelementptr %605[%646, %646] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %648 = llvm.load %647 : !llvm<"i32*">
    %649 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %650 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %651 = llvm.getelementptr %605[%649, %650] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %652 = llvm.load %651 : !llvm<"i64*">
    %653 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %654 = llvm.insertvalue %648, %653[0] : !llvm<"{ i32, i64 }">
    %655 = llvm.insertvalue %652, %654[1] : !llvm<"{ i32, i64 }">
    %656 = llvm.call @lua_le(%645, %655) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }">
    %657 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %658 = llvm.alloca %657 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %659 = llvm.extractvalue %656[0] : !llvm<"{ i32, i64 }">
    %660 = llvm.extractvalue %656[1] : !llvm<"{ i32, i64 }">
    %661 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %662 = llvm.getelementptr %658[%661, %661] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %659, %662 : !llvm<"i32*">
    %663 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %664 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %665 = llvm.getelementptr %658[%663, %664] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %660, %665 : !llvm<"i64*">
    %666 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %667 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %668 = llvm.getelementptr %658[%666, %667] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %669 = llvm.load %668 : !llvm<"i64*">
    %670 = llvm.trunc %669 : !llvm.i64 to !llvm.i1
    llvm.cond_br %670, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %671 = llvm.mlir.addressof @g_arg_pack_ptr : !llvm<"i64*">
    %672 = llvm.bitcast %671 : !llvm<"i64*"> to !llvm<"i8**">
    %673 = llvm.load %672 : !llvm<"i8**">
    %674 = llvm.call @realloc(%673, %14) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    llvm.store %674, %672 : !llvm<"i8**">
    %675 = llvm.mlir.undef : !llvm<"{ i32, { i32, i64 }* }">
    %676 = llvm.bitcast %674 : !llvm<"i8*"> to !llvm<"{ i32, i64 }*">
    %677 = llvm.insertvalue %10, %675[0] : !llvm<"{ i32, { i32, i64 }* }">
    %678 = llvm.insertvalue %676, %677[1] : !llvm<"{ i32, { i32, i64 }* }">
    %679 = llvm.extractvalue %678[1] : !llvm<"{ i32, { i32, i64 }* }">
    %680 = llvm.getelementptr %679[%12] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %681 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %682 = llvm.getelementptr %116[%681, %681] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %683 = llvm.load %682 : !llvm<"i32*">
    %684 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %685 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %686 = llvm.getelementptr %116[%684, %685] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %687 = llvm.load %686 : !llvm<"i64*">
    %688 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %689 = llvm.insertvalue %683, %688[0] : !llvm<"{ i32, i64 }">
    %690 = llvm.insertvalue %687, %689[1] : !llvm<"{ i32, i64 }">
    llvm.store %690, %680 : !llvm<"{ i32, i64 }*">
    %691 = llvm.extractvalue %678[1] : !llvm<"{ i32, { i32, i64 }* }">
    %692 = llvm.getelementptr %691[%9] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %693 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %694 = llvm.getelementptr %480[%693, %693] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %695 = llvm.load %694 : !llvm<"i32*">
    %696 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %697 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %698 = llvm.getelementptr %480[%696, %697] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %699 = llvm.load %698 : !llvm<"i64*">
    %700 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %701 = llvm.insertvalue %695, %700[0] : !llvm<"{ i32, i64 }">
    %702 = llvm.insertvalue %699, %701[1] : !llvm<"{ i32, i64 }">
    llvm.store %702, %692 : !llvm<"{ i32, i64 }*">
    %703 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %704 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %705 = llvm.getelementptr %39[%703, %704] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %706 = llvm.bitcast %705 : !llvm<"i64*"> to !llvm<"i8**">
    %707 = llvm.load %706 : !llvm<"i8**">
    %708 = llvm.bitcast %707 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %709 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %710 = llvm.getelementptr %708[%709, %709] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %711 = llvm.load %710 : !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %712 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %713 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %714 = llvm.getelementptr %39[%712, %713] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %715 = llvm.bitcast %714 : !llvm<"i64*"> to !llvm<"i8**">
    %716 = llvm.load %715 : !llvm<"i8**">
    %717 = llvm.bitcast %716 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %718 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %719 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %720 = llvm.getelementptr %717[%718, %719] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, i64 }***">
    %721 = llvm.load %720 : !llvm<"{ i32, i64 }***">
    %722 = llvm.call %711(%721, %678) : (!llvm<"{ i32, i64 }**">, !llvm<"{ i32, { i32, i64 }* }">) -> !llvm<"{ i32, { i32, i64 }* }">
    %723 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %724 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %725 = llvm.getelementptr %30[%723, %724] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %726 = llvm.bitcast %725 : !llvm<"i64*"> to !llvm<"i8**">
    %727 = llvm.load %726 : !llvm<"i8**">
    %728 = llvm.bitcast %727 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %729 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %730 = llvm.getelementptr %728[%729, %729] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %731 = llvm.load %730 : !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %732 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %733 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %734 = llvm.getelementptr %30[%732, %733] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %735 = llvm.bitcast %734 : !llvm<"i64*"> to !llvm<"i8**">
    %736 = llvm.load %735 : !llvm<"i8**">
    %737 = llvm.bitcast %736 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %738 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %739 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %740 = llvm.getelementptr %737[%738, %739] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, i64 }***">
    %741 = llvm.load %740 : !llvm<"{ i32, i64 }***">
    %742 = llvm.call %731(%741, %722) : (!llvm<"{ i32, i64 }**">, !llvm<"{ i32, { i32, i64 }* }">) -> !llvm<"{ i32, { i32, i64 }* }">
    %743 = llvm.extractvalue %742[1] : !llvm<"{ i32, { i32, i64 }* }">
    %744 = llvm.getelementptr %743[%12] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %745 = llvm.load %744 : !llvm<"{ i32, i64 }*">
    %746 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %747 = llvm.alloca %746 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %748 = llvm.extractvalue %745[0] : !llvm<"{ i32, i64 }">
    %749 = llvm.extractvalue %745[1] : !llvm<"{ i32, i64 }">
    %750 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %751 = llvm.getelementptr %747[%750, %750] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %748, %751 : !llvm<"i32*">
    %752 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %753 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %754 = llvm.getelementptr %747[%752, %753] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %749, %754 : !llvm<"i64*">
    %755 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %756 = llvm.getelementptr %614[%755, %755] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %757 = llvm.load %756 : !llvm<"i32*">
    %758 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %759 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %760 = llvm.getelementptr %614[%758, %759] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %761 = llvm.load %760 : !llvm<"i64*">
    %762 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %763 = llvm.insertvalue %757, %762[0] : !llvm<"{ i32, i64 }">
    %764 = llvm.insertvalue %761, %763[1] : !llvm<"{ i32, i64 }">
    %765 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %766 = llvm.getelementptr %747[%765, %765] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %767 = llvm.load %766 : !llvm<"i32*">
    %768 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %769 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %770 = llvm.getelementptr %747[%768, %769] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %771 = llvm.load %770 : !llvm<"i64*">
    %772 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %773 = llvm.insertvalue %767, %772[0] : !llvm<"{ i32, i64 }">
    %774 = llvm.insertvalue %771, %773[1] : !llvm<"{ i32, i64 }">
    %775 = llvm.call @lua_add(%764, %774) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }">
    %776 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %777 = llvm.alloca %776 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %778 = llvm.extractvalue %775[0] : !llvm<"{ i32, i64 }">
    %779 = llvm.extractvalue %775[1] : !llvm<"{ i32, i64 }">
    %780 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %781 = llvm.getelementptr %777[%780, %780] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %778, %781 : !llvm<"i32*">
    %782 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %783 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %784 = llvm.getelementptr %777[%782, %783] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %779, %784 : !llvm<"i64*">
    %785 = llvm.mlir.addressof @g_arg_pack_ptr : !llvm<"i64*">
    %786 = llvm.bitcast %785 : !llvm<"i64*"> to !llvm<"i8**">
    %787 = llvm.load %786 : !llvm<"i8**">
    %788 = llvm.call @realloc(%787, %14) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    llvm.store %788, %786 : !llvm<"i8**">
    %789 = llvm.mlir.undef : !llvm<"{ i32, { i32, i64 }* }">
    %790 = llvm.bitcast %788 : !llvm<"i8*"> to !llvm<"{ i32, i64 }*">
    %791 = llvm.insertvalue %10, %789[0] : !llvm<"{ i32, { i32, i64 }* }">
    %792 = llvm.insertvalue %790, %791[1] : !llvm<"{ i32, { i32, i64 }* }">
    %793 = llvm.extractvalue %792[1] : !llvm<"{ i32, { i32, i64 }* }">
    %794 = llvm.getelementptr %793[%12] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %795 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %796 = llvm.getelementptr %447[%795, %795] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %797 = llvm.load %796 : !llvm<"i32*">
    %798 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %799 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %800 = llvm.getelementptr %447[%798, %799] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %801 = llvm.load %800 : !llvm<"i64*">
    %802 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %803 = llvm.insertvalue %797, %802[0] : !llvm<"{ i32, i64 }">
    %804 = llvm.insertvalue %801, %803[1] : !llvm<"{ i32, i64 }">
    llvm.store %804, %794 : !llvm<"{ i32, i64 }*">
    %805 = llvm.extractvalue %792[1] : !llvm<"{ i32, { i32, i64 }* }">
    %806 = llvm.getelementptr %805[%9] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %807 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %808 = llvm.getelementptr %480[%807, %807] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %809 = llvm.load %808 : !llvm<"i32*">
    %810 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %811 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %812 = llvm.getelementptr %480[%810, %811] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %813 = llvm.load %812 : !llvm<"i64*">
    %814 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %815 = llvm.insertvalue %809, %814[0] : !llvm<"{ i32, i64 }">
    %816 = llvm.insertvalue %813, %815[1] : !llvm<"{ i32, i64 }">
    llvm.store %816, %806 : !llvm<"{ i32, i64 }*">
    %817 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %818 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %819 = llvm.getelementptr %39[%817, %818] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %820 = llvm.bitcast %819 : !llvm<"i64*"> to !llvm<"i8**">
    %821 = llvm.load %820 : !llvm<"i8**">
    %822 = llvm.bitcast %821 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %823 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %824 = llvm.getelementptr %822[%823, %823] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %825 = llvm.load %824 : !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %826 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %827 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %828 = llvm.getelementptr %39[%826, %827] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %829 = llvm.bitcast %828 : !llvm<"i64*"> to !llvm<"i8**">
    %830 = llvm.load %829 : !llvm<"i8**">
    %831 = llvm.bitcast %830 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %832 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %833 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %834 = llvm.getelementptr %831[%832, %833] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, i64 }***">
    %835 = llvm.load %834 : !llvm<"{ i32, i64 }***">
    %836 = llvm.call %825(%835, %792) : (!llvm<"{ i32, i64 }**">, !llvm<"{ i32, { i32, i64 }* }">) -> !llvm<"{ i32, { i32, i64 }* }">
    %837 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %838 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %839 = llvm.getelementptr %30[%837, %838] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %840 = llvm.bitcast %839 : !llvm<"i64*"> to !llvm<"i8**">
    %841 = llvm.load %840 : !llvm<"i8**">
    %842 = llvm.bitcast %841 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %843 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %844 = llvm.getelementptr %842[%843, %843] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %845 = llvm.load %844 : !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %846 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %847 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %848 = llvm.getelementptr %30[%846, %847] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %849 = llvm.bitcast %848 : !llvm<"i64*"> to !llvm<"i8**">
    %850 = llvm.load %849 : !llvm<"i8**">
    %851 = llvm.bitcast %850 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %852 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %853 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %854 = llvm.getelementptr %851[%852, %853] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, i64 }***">
    %855 = llvm.load %854 : !llvm<"{ i32, i64 }***">
    %856 = llvm.call %845(%855, %836) : (!llvm<"{ i32, i64 }**">, !llvm<"{ i32, { i32, i64 }* }">) -> !llvm<"{ i32, { i32, i64 }* }">
    %857 = llvm.extractvalue %856[1] : !llvm<"{ i32, { i32, i64 }* }">
    %858 = llvm.getelementptr %857[%12] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %859 = llvm.load %858 : !llvm<"{ i32, i64 }*">
    %860 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %861 = llvm.alloca %860 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %862 = llvm.extractvalue %859[0] : !llvm<"{ i32, i64 }">
    %863 = llvm.extractvalue %859[1] : !llvm<"{ i32, i64 }">
    %864 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %865 = llvm.getelementptr %861[%864, %864] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %862, %865 : !llvm<"i32*">
    %866 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %867 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %868 = llvm.getelementptr %861[%866, %867] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %863, %868 : !llvm<"i64*">
    %869 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %870 = llvm.getelementptr %777[%869, %869] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %871 = llvm.load %870 : !llvm<"i32*">
    %872 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %873 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %874 = llvm.getelementptr %777[%872, %873] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %875 = llvm.load %874 : !llvm<"i64*">
    %876 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %877 = llvm.insertvalue %871, %876[0] : !llvm<"{ i32, i64 }">
    %878 = llvm.insertvalue %875, %877[1] : !llvm<"{ i32, i64 }">
    %879 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %880 = llvm.getelementptr %861[%879, %879] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %881 = llvm.load %880 : !llvm<"i32*">
    %882 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %883 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %884 = llvm.getelementptr %861[%882, %883] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %885 = llvm.load %884 : !llvm<"i64*">
    %886 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %887 = llvm.insertvalue %881, %886[0] : !llvm<"{ i32, i64 }">
    %888 = llvm.insertvalue %885, %887[1] : !llvm<"{ i32, i64 }">
    %889 = llvm.call @lua_add(%878, %888) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }">
    %890 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %891 = llvm.alloca %890 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %892 = llvm.extractvalue %889[0] : !llvm<"{ i32, i64 }">
    %893 = llvm.extractvalue %889[1] : !llvm<"{ i32, i64 }">
    %894 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %895 = llvm.getelementptr %891[%894, %894] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %892, %895 : !llvm<"i32*">
    %896 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %897 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %898 = llvm.getelementptr %891[%896, %897] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %893, %898 : !llvm<"i64*">
    %899 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %900 = llvm.getelementptr %891[%899, %899] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %901 = llvm.load %900 : !llvm<"i32*">
    %902 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %903 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %904 = llvm.getelementptr %891[%902, %903] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %905 = llvm.load %904 : !llvm<"i64*">
    %906 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %907 = llvm.getelementptr %614[%906, %906] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %901, %907 : !llvm<"i32*">
    %908 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %909 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %910 = llvm.getelementptr %614[%908, %909] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %905, %910 : !llvm<"i64*">
    %911 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %912 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %913 = llvm.getelementptr %628[%911, %912] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %914 = llvm.load %913 : !llvm<"i64*">
    %915 = llvm.bitcast %914 : !llvm.i64 to !llvm.double
    %916 = llvm.fadd %915, %626 : !llvm.double
    %917 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %918 = llvm.alloca %917 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %919 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %920 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %921 = llvm.getelementptr %918[%920, %920] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %919, %921 : !llvm<"i32*">
    %922 = llvm.bitcast %916 : !llvm.double to !llvm.i64
    %923 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %924 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %925 = llvm.getelementptr %918[%923, %924] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %922, %925 : !llvm<"i64*">
    %926 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %927 = llvm.getelementptr %918[%926, %926] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %928 = llvm.load %927 : !llvm<"i32*">
    %929 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %930 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %931 = llvm.getelementptr %918[%929, %930] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %932 = llvm.load %931 : !llvm<"i64*">
    %933 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %934 = llvm.getelementptr %628[%933, %933] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %928, %934 : !llvm<"i32*">
    %935 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %936 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %937 = llvm.getelementptr %628[%935, %936] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %932, %937 : !llvm<"i64*">
    llvm.br ^bb3
  ^bb5:  // pred: ^bb3
    %938 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %939 = llvm.getelementptr %605[%938, %938] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %940 = llvm.load %939 : !llvm<"i32*">
    %941 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %942 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %943 = llvm.getelementptr %605[%941, %942] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %944 = llvm.load %943 : !llvm<"i64*">
    %945 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %946 = llvm.insertvalue %940, %945[0] : !llvm<"{ i32, i64 }">
    %947 = llvm.insertvalue %944, %946[1] : !llvm<"{ i32, i64 }">
    %948 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %949 = llvm.getelementptr %427[%948, %948] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %950 = llvm.load %949 : !llvm<"i32*">
    %951 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %952 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %953 = llvm.getelementptr %427[%951, %952] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %954 = llvm.load %953 : !llvm<"i64*">
    %955 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %956 = llvm.insertvalue %950, %955[0] : !llvm<"{ i32, i64 }">
    %957 = llvm.insertvalue %954, %956[1] : !llvm<"{ i32, i64 }">
    %958 = llvm.call @lua_mul(%947, %957) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }">
    %959 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %960 = llvm.alloca %959 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %961 = llvm.extractvalue %958[0] : !llvm<"{ i32, i64 }">
    %962 = llvm.extractvalue %958[1] : !llvm<"{ i32, i64 }">
    %963 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %964 = llvm.getelementptr %960[%963, %963] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %961, %964 : !llvm<"i32*">
    %965 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %966 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %967 = llvm.getelementptr %960[%965, %966] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %962, %967 : !llvm<"i64*">
    %968 = llvm.mlir.addressof @g_arg_pack_ptr : !llvm<"i64*">
    %969 = llvm.bitcast %968 : !llvm<"i64*"> to !llvm<"i8**">
    %970 = llvm.load %969 : !llvm<"i8**">
    %971 = llvm.call @realloc(%970, %15) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    llvm.store %971, %969 : !llvm<"i8**">
    %972 = llvm.mlir.undef : !llvm<"{ i32, { i32, i64 }* }">
    %973 = llvm.bitcast %971 : !llvm<"i8*"> to !llvm<"{ i32, i64 }*">
    %974 = llvm.insertvalue %7, %972[0] : !llvm<"{ i32, { i32, i64 }* }">
    %975 = llvm.insertvalue %973, %974[1] : !llvm<"{ i32, { i32, i64 }* }">
    %976 = llvm.extractvalue %975[1] : !llvm<"{ i32, { i32, i64 }* }">
    %977 = llvm.getelementptr %976[%12] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %978 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %979 = llvm.getelementptr %960[%978, %978] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %980 = llvm.load %979 : !llvm<"i32*">
    %981 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %982 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %983 = llvm.getelementptr %960[%981, %982] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %984 = llvm.load %983 : !llvm<"i64*">
    %985 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %986 = llvm.insertvalue %980, %985[0] : !llvm<"{ i32, i64 }">
    %987 = llvm.insertvalue %984, %986[1] : !llvm<"{ i32, i64 }">
    llvm.store %987, %977 : !llvm<"{ i32, i64 }*">
    %988 = llvm.extractvalue %975[1] : !llvm<"{ i32, { i32, i64 }* }">
    %989 = llvm.getelementptr %988[%9] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %990 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %991 = llvm.getelementptr %456[%990, %990] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %992 = llvm.load %991 : !llvm<"i32*">
    %993 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %994 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %995 = llvm.getelementptr %456[%993, %994] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %996 = llvm.load %995 : !llvm<"i64*">
    %997 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %998 = llvm.insertvalue %992, %997[0] : !llvm<"{ i32, i64 }">
    %999 = llvm.insertvalue %996, %998[1] : !llvm<"{ i32, i64 }">
    llvm.store %999, %989 : !llvm<"{ i32, i64 }*">
    %1000 = llvm.extractvalue %975[1] : !llvm<"{ i32, { i32, i64 }* }">
    %1001 = llvm.getelementptr %1000[%10] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %1002 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1003 = llvm.getelementptr %480[%1002, %1002] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %1004 = llvm.load %1003 : !llvm<"i32*">
    %1005 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1006 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1007 = llvm.getelementptr %480[%1005, %1006] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %1008 = llvm.load %1007 : !llvm<"i64*">
    %1009 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %1010 = llvm.insertvalue %1004, %1009[0] : !llvm<"{ i32, i64 }">
    %1011 = llvm.insertvalue %1008, %1010[1] : !llvm<"{ i32, i64 }">
    llvm.store %1011, %1001 : !llvm<"{ i32, i64 }*">
    %1012 = llvm.extractvalue %975[1] : !llvm<"{ i32, { i32, i64 }* }">
    %1013 = llvm.getelementptr %1012[%11] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %1014 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1015 = llvm.getelementptr %242[%1014, %1014] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %1016 = llvm.load %1015 : !llvm<"i32*">
    %1017 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1018 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1019 = llvm.getelementptr %242[%1017, %1018] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %1020 = llvm.load %1019 : !llvm<"i64*">
    %1021 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %1022 = llvm.insertvalue %1016, %1021[0] : !llvm<"{ i32, i64 }">
    %1023 = llvm.insertvalue %1020, %1022[1] : !llvm<"{ i32, i64 }">
    llvm.store %1023, %1013 : !llvm<"{ i32, i64 }*">
    %1024 = llvm.extractvalue %975[1] : !llvm<"{ i32, { i32, i64 }* }">
    %1025 = llvm.getelementptr %1024[%8] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %1026 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1027 = llvm.getelementptr %614[%1026, %1026] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %1028 = llvm.load %1027 : !llvm<"i32*">
    %1029 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1030 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1031 = llvm.getelementptr %614[%1029, %1030] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %1032 = llvm.load %1031 : !llvm<"i64*">
    %1033 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %1034 = llvm.insertvalue %1028, %1033[0] : !llvm<"{ i32, i64 }">
    %1035 = llvm.insertvalue %1032, %1034[1] : !llvm<"{ i32, i64 }">
    llvm.store %1035, %1025 : !llvm<"{ i32, i64 }*">
    %1036 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1037 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1038 = llvm.getelementptr %21[%1036, %1037] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %1039 = llvm.bitcast %1038 : !llvm<"i64*"> to !llvm<"i8**">
    %1040 = llvm.load %1039 : !llvm<"i8**">
    %1041 = llvm.bitcast %1040 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %1042 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1043 = llvm.getelementptr %1041[%1042, %1042] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %1044 = llvm.load %1043 : !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %1045 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1046 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1047 = llvm.getelementptr %21[%1045, %1046] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %1048 = llvm.bitcast %1047 : !llvm<"i64*"> to !llvm<"i8**">
    %1049 = llvm.load %1048 : !llvm<"i8**">
    %1050 = llvm.bitcast %1049 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %1051 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1052 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1053 = llvm.getelementptr %1050[%1051, %1052] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, i64 }***">
    %1054 = llvm.load %1053 : !llvm<"{ i32, i64 }***">
    %1055 = llvm.call %1044(%1054, %975) : (!llvm<"{ i32, i64 }**">, !llvm<"{ i32, { i32, i64 }* }">) -> !llvm<"{ i32, { i32, i64 }* }">
    %1056 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1057 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1058 = llvm.getelementptr %480[%1056, %1057] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %1059 = llvm.load %1058 : !llvm<"i64*">
    %1060 = llvm.bitcast %1059 : !llvm.i64 to !llvm.double
    %1061 = llvm.fadd %1060, %473 : !llvm.double
    %1062 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1063 = llvm.alloca %1062 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %1064 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %1065 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1066 = llvm.getelementptr %1063[%1065, %1065] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %1064, %1066 : !llvm<"i32*">
    %1067 = llvm.bitcast %1061 : !llvm.double to !llvm.i64
    %1068 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1069 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1070 = llvm.getelementptr %1063[%1068, %1069] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %1067, %1070 : !llvm<"i64*">
    %1071 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1072 = llvm.getelementptr %1063[%1071, %1071] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %1073 = llvm.load %1072 : !llvm<"i32*">
    %1074 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1075 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1076 = llvm.getelementptr %1063[%1074, %1075] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %1077 = llvm.load %1076 : !llvm<"i64*">
    %1078 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1079 = llvm.getelementptr %480[%1078, %1078] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %1073, %1079 : !llvm<"i32*">
    %1080 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1081 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1082 = llvm.getelementptr %480[%1080, %1081] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %1077, %1082 : !llvm<"i64*">
    llvm.br ^bb1
  ^bb6:  // pred: ^bb1
    %1083 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1084 = llvm.alloca %1083 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %1085 = llvm.mlir.constant(3 : i32) : !llvm.i32
    %1086 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1087 = llvm.getelementptr %1084[%1086, %1086] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %1085, %1087 : !llvm<"i32*">
    %1088 = llvm.mlir.addressof @lua_anon_string_3 : !llvm<"[24 x i8]*">
    %1089 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1090 = llvm.getelementptr %1088[%1089, %1089] : (!llvm<"[24 x i8]*">, !llvm.i32, !llvm.i32) -> !llvm<"i8*">
    %1091 = llvm.mlir.constant(24 : i64) : !llvm.i64
    %1092 = llvm.call @lua_load_string_impl(%1090, %1091) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    %1093 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1094 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1095 = llvm.getelementptr %1084[%1093, %1094] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %1096 = llvm.bitcast %1095 : !llvm<"i64*"> to !llvm<"i8**">
    llvm.store %1092, %1096 : !llvm<"i8**">
    %1097 = llvm.mlir.addressof @g_arg_pack_ptr : !llvm<"i64*">
    %1098 = llvm.bitcast %1097 : !llvm<"i64*"> to !llvm<"i8**">
    %1099 = llvm.load %1098 : !llvm<"i8**">
    %1100 = llvm.call @realloc(%1099, %16) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    llvm.store %1100, %1098 : !llvm<"i8**">
    %1101 = llvm.mlir.undef : !llvm<"{ i32, { i32, i64 }* }">
    %1102 = llvm.bitcast %1100 : !llvm<"i8*"> to !llvm<"{ i32, i64 }*">
    %1103 = llvm.insertvalue %9, %1101[0] : !llvm<"{ i32, { i32, i64 }* }">
    %1104 = llvm.insertvalue %1102, %1103[1] : !llvm<"{ i32, { i32, i64 }* }">
    %1105 = llvm.extractvalue %1104[1] : !llvm<"{ i32, { i32, i64 }* }">
    %1106 = llvm.getelementptr %1105[%12] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %1107 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1108 = llvm.getelementptr %418[%1107, %1107] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %1109 = llvm.load %1108 : !llvm<"i32*">
    %1110 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1111 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1112 = llvm.getelementptr %418[%1110, %1111] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %1113 = llvm.load %1112 : !llvm<"i64*">
    %1114 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %1115 = llvm.insertvalue %1109, %1114[0] : !llvm<"{ i32, i64 }">
    %1116 = llvm.insertvalue %1113, %1115[1] : !llvm<"{ i32, i64 }">
    llvm.store %1116, %1106 : !llvm<"{ i32, i64 }*">
    %1117 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1118 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1119 = llvm.getelementptr %30[%1117, %1118] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %1120 = llvm.bitcast %1119 : !llvm<"i64*"> to !llvm<"i8**">
    %1121 = llvm.load %1120 : !llvm<"i8**">
    %1122 = llvm.bitcast %1121 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %1123 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1124 = llvm.getelementptr %1122[%1123, %1123] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %1125 = llvm.load %1124 : !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %1126 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1127 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1128 = llvm.getelementptr %30[%1126, %1127] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %1129 = llvm.bitcast %1128 : !llvm<"i64*"> to !llvm<"i8**">
    %1130 = llvm.load %1129 : !llvm<"i8**">
    %1131 = llvm.bitcast %1130 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %1132 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1133 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1134 = llvm.getelementptr %1131[%1132, %1133] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, i64 }***">
    %1135 = llvm.load %1134 : !llvm<"{ i32, i64 }***">
    %1136 = llvm.call %1125(%1135, %1104) : (!llvm<"{ i32, i64 }**">, !llvm<"{ i32, { i32, i64 }* }">) -> !llvm<"{ i32, { i32, i64 }* }">
    %1137 = llvm.extractvalue %1136[0] : !llvm<"{ i32, { i32, i64 }* }">
    %1138 = llvm.add %1137, %11 : !llvm.i32
    %1139 = llvm.mlir.addressof @g_arg_pack_ptr : !llvm<"i64*">
    %1140 = llvm.bitcast %1139 : !llvm<"i64*"> to !llvm<"i8**">
    %1141 = llvm.load %1140 : !llvm<"i8**">
    %1142 = llvm.sext %1138 : !llvm.i32 to !llvm.i64
    %1143 = llvm.call @realloc(%1141, %1142) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    llvm.store %1143, %1140 : !llvm<"i8**">
    %1144 = llvm.mlir.undef : !llvm<"{ i32, { i32, i64 }* }">
    %1145 = llvm.bitcast %1143 : !llvm<"i8*"> to !llvm<"{ i32, i64 }*">
    %1146 = llvm.insertvalue %1138, %1144[0] : !llvm<"{ i32, { i32, i64 }* }">
    %1147 = llvm.insertvalue %1145, %1146[1] : !llvm<"{ i32, { i32, i64 }* }">
    %1148 = llvm.extractvalue %1147[1] : !llvm<"{ i32, { i32, i64 }* }">
    %1149 = llvm.getelementptr %1148[%12] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %1150 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1151 = llvm.getelementptr %1084[%1150, %1150] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %1152 = llvm.load %1151 : !llvm<"i32*">
    %1153 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1154 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1155 = llvm.getelementptr %1084[%1153, %1154] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %1156 = llvm.load %1155 : !llvm<"i64*">
    %1157 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %1158 = llvm.insertvalue %1152, %1157[0] : !llvm<"{ i32, i64 }">
    %1159 = llvm.insertvalue %1156, %1158[1] : !llvm<"{ i32, i64 }">
    llvm.store %1159, %1149 : !llvm<"{ i32, i64 }*">
    %1160 = llvm.extractvalue %1147[1] : !llvm<"{ i32, { i32, i64 }* }">
    %1161 = llvm.getelementptr %1160[%9] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %1162 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1163 = llvm.getelementptr %107[%1162, %1162] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %1164 = llvm.load %1163 : !llvm<"i32*">
    %1165 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1166 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1167 = llvm.getelementptr %107[%1165, %1166] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %1168 = llvm.load %1167 : !llvm<"i64*">
    %1169 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %1170 = llvm.insertvalue %1164, %1169[0] : !llvm<"{ i32, i64 }">
    %1171 = llvm.insertvalue %1168, %1170[1] : !llvm<"{ i32, i64 }">
    llvm.store %1171, %1161 : !llvm<"{ i32, i64 }*">
    %1172 = llvm.extractvalue %1147[1] : !llvm<"{ i32, { i32, i64 }* }">
    %1173 = llvm.getelementptr %1172[%10] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %1174 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1175 = llvm.getelementptr %242[%1174, %1174] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %1176 = llvm.load %1175 : !llvm<"i32*">
    %1177 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1178 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1179 = llvm.getelementptr %242[%1177, %1178] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %1180 = llvm.load %1179 : !llvm<"i64*">
    %1181 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %1182 = llvm.insertvalue %1176, %1181[0] : !llvm<"{ i32, i64 }">
    %1183 = llvm.insertvalue %1180, %1182[1] : !llvm<"{ i32, i64 }">
    llvm.store %1183, %1173 : !llvm<"{ i32, i64 }*">
    llvm.call @lua_pack_insert_all(%1147, %1136, %11) : (!llvm<"{ i32, { i32, i64 }* }">, !llvm<"{ i32, { i32, i64 }* }">, !llvm.i32) -> ()
    %1184 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1185 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1186 = llvm.getelementptr %21[%1184, %1185] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %1187 = llvm.bitcast %1186 : !llvm<"i64*"> to !llvm<"i8**">
    %1188 = llvm.load %1187 : !llvm<"i8**">
    %1189 = llvm.bitcast %1188 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %1190 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1191 = llvm.getelementptr %1189[%1190, %1190] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %1192 = llvm.load %1191 : !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %1193 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1194 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1195 = llvm.getelementptr %21[%1193, %1194] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %1196 = llvm.bitcast %1195 : !llvm<"i64*"> to !llvm<"i8**">
    %1197 = llvm.load %1196 : !llvm<"i8**">
    %1198 = llvm.bitcast %1197 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %1199 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1200 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1201 = llvm.getelementptr %1198[%1199, %1200] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, i64 }***">
    %1202 = llvm.load %1201 : !llvm<"{ i32, i64 }***">
    %1203 = llvm.call %1192(%1202, %1147) : (!llvm<"{ i32, i64 }**">, !llvm<"{ i32, { i32, i64 }* }">) -> !llvm<"{ i32, { i32, i64 }* }">
    %1204 = llvm.mlir.addressof @g_ret_pack_ptr : !llvm<"i64*">
    %1205 = llvm.bitcast %1204 : !llvm<"i64*"> to !llvm<"i8**">
    %1206 = llvm.load %1205 : !llvm<"i8**">
    %1207 = llvm.call @realloc(%1206, %17) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    llvm.store %1207, %1205 : !llvm<"i8**">
    %1208 = llvm.mlir.undef : !llvm<"{ i32, { i32, i64 }* }">
    %1209 = llvm.bitcast %1207 : !llvm<"i8*"> to !llvm<"{ i32, i64 }*">
    %1210 = llvm.insertvalue %12, %1208[0] : !llvm<"{ i32, { i32, i64 }* }">
    %1211 = llvm.insertvalue %1209, %1210[1] : !llvm<"{ i32, { i32, i64 }* }">
    llvm.return %1211 : !llvm<"{ i32, { i32, i64 }* }">
  }
  llvm.func @lua_anon_fcn_0(%arg0: !llvm<"{ i32, i64 }**">, %arg1: !llvm<"{ i32, { i32, i64 }* }">) -> !llvm<"{ i32, { i32, i64 }* }"> {
    %0 = llvm.mlir.constant(1 : i64) : !llvm.i64
    %1 = llvm.mlir.constant(2 : i64) : !llvm.i64
    %2 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %3 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %5 = llvm.mlir.constant(1 : index) : !llvm.i64
    %6 = llvm.getelementptr %arg0[%4] : (!llvm<"{ i32, i64 }**">, !llvm.i32) -> !llvm<"{ i32, i64 }**">
    %7 = llvm.load %6 : !llvm<"{ i32, i64 }**">
    %8 = llvm.call @lua_pack_get(%arg1, %4) : (!llvm<"{ i32, { i32, i64 }* }">, !llvm.i32) -> !llvm<"{ i32, i64 }">
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.alloca %9 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %11 = llvm.extractvalue %8[0] : !llvm<"{ i32, i64 }">
    %12 = llvm.extractvalue %8[1] : !llvm<"{ i32, i64 }">
    %13 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %14 = llvm.getelementptr %10[%13, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %11, %14 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %10[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %12, %17 : !llvm<"i64*">
    %18 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %19 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %20 = llvm.getelementptr %10[%18, %19] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %21 = llvm.bitcast %20 : !llvm<"i64*"> to !llvm<"i8**">
    %22 = llvm.load %21 : !llvm<"i8**">
    %23 = llvm.call @lua_table_get_prealloc_impl(%22, %0) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"{ i32, i64 }">
    %24 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %25 = llvm.alloca %24 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %26 = llvm.extractvalue %23[0] : !llvm<"{ i32, i64 }">
    %27 = llvm.extractvalue %23[1] : !llvm<"{ i32, i64 }">
    %28 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %29 = llvm.getelementptr %25[%28, %28] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %26, %29 : !llvm<"i32*">
    %30 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %31 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %32 = llvm.getelementptr %25[%30, %31] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %27, %32 : !llvm<"i64*">
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.getelementptr %25[%33, %33] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %35 = llvm.load %34 : !llvm<"i32*">
    %36 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %37 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %38 = llvm.getelementptr %25[%36, %37] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %39 = llvm.load %38 : !llvm<"i64*">
    %40 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %41 = llvm.insertvalue %35, %40[0] : !llvm<"{ i32, i64 }">
    %42 = llvm.insertvalue %39, %41[1] : !llvm<"{ i32, i64 }">
    %43 = llvm.call @lua_convert_bool_like(%42) : (!llvm<"{ i32, i64 }">) -> !llvm.i1
    llvm.cond_br %43, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %44 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %45 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %46 = llvm.getelementptr %10[%44, %45] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %47 = llvm.bitcast %46 : !llvm<"i64*"> to !llvm<"i8**">
    %48 = llvm.load %47 : !llvm<"i8**">
    %49 = llvm.call @lua_table_get_prealloc_impl(%48, %2) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"{ i32, i64 }">
    %50 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %51 = llvm.alloca %50 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %52 = llvm.extractvalue %49[0] : !llvm<"{ i32, i64 }">
    %53 = llvm.extractvalue %49[1] : !llvm<"{ i32, i64 }">
    %54 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %55 = llvm.getelementptr %51[%54, %54] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %52, %55 : !llvm<"i32*">
    %56 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %57 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %58 = llvm.getelementptr %51[%56, %57] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %53, %58 : !llvm<"i64*">
    %59 = llvm.mlir.addressof @g_arg_pack_ptr : !llvm<"i64*">
    %60 = llvm.bitcast %59 : !llvm<"i64*"> to !llvm<"i8**">
    %61 = llvm.load %60 : !llvm<"i8**">
    %62 = llvm.call @realloc(%61, %5) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    llvm.store %62, %60 : !llvm<"i8**">
    %63 = llvm.mlir.undef : !llvm<"{ i32, { i32, i64 }* }">
    %64 = llvm.bitcast %62 : !llvm<"i8*"> to !llvm<"{ i32, i64 }*">
    %65 = llvm.insertvalue %3, %63[0] : !llvm<"{ i32, { i32, i64 }* }">
    %66 = llvm.insertvalue %64, %65[1] : !llvm<"{ i32, { i32, i64 }* }">
    %67 = llvm.extractvalue %66[1] : !llvm<"{ i32, { i32, i64 }* }">
    %68 = llvm.getelementptr %67[%4] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %69 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %70 = llvm.getelementptr %25[%69, %69] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %71 = llvm.load %70 : !llvm<"i32*">
    %72 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %73 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %74 = llvm.getelementptr %25[%72, %73] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %75 = llvm.load %74 : !llvm<"i64*">
    %76 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %77 = llvm.insertvalue %71, %76[0] : !llvm<"{ i32, i64 }">
    %78 = llvm.insertvalue %75, %77[1] : !llvm<"{ i32, i64 }">
    llvm.store %78, %68 : !llvm<"{ i32, i64 }*">
    %79 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %80 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %81 = llvm.getelementptr %7[%79, %80] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %82 = llvm.bitcast %81 : !llvm<"i64*"> to !llvm<"i8**">
    %83 = llvm.load %82 : !llvm<"i8**">
    %84 = llvm.bitcast %83 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %85 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %86 = llvm.getelementptr %84[%85, %85] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %87 = llvm.load %86 : !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %88 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %89 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %90 = llvm.getelementptr %7[%88, %89] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %91 = llvm.bitcast %90 : !llvm<"i64*"> to !llvm<"i8**">
    %92 = llvm.load %91 : !llvm<"i8**">
    %93 = llvm.bitcast %92 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %94 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %95 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %96 = llvm.getelementptr %93[%94, %95] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, i64 }***">
    %97 = llvm.load %96 : !llvm<"{ i32, i64 }***">
    %98 = llvm.call %87(%97, %66) : (!llvm<"{ i32, i64 }**">, !llvm<"{ i32, { i32, i64 }* }">) -> !llvm<"{ i32, { i32, i64 }* }">
    %99 = llvm.extractvalue %98[1] : !llvm<"{ i32, { i32, i64 }* }">
    %100 = llvm.getelementptr %99[%4] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %101 = llvm.load %100 : !llvm<"{ i32, i64 }*">
    %102 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %103 = llvm.alloca %102 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %104 = llvm.extractvalue %101[0] : !llvm<"{ i32, i64 }">
    %105 = llvm.extractvalue %101[1] : !llvm<"{ i32, i64 }">
    %106 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %107 = llvm.getelementptr %103[%106, %106] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %104, %107 : !llvm<"i32*">
    %108 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %109 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %110 = llvm.getelementptr %103[%108, %109] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %105, %110 : !llvm<"i64*">
    %111 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %112 = llvm.getelementptr %51[%111, %111] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %113 = llvm.load %112 : !llvm<"i32*">
    %114 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %115 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %116 = llvm.getelementptr %51[%114, %115] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %117 = llvm.load %116 : !llvm<"i64*">
    %118 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %119 = llvm.insertvalue %113, %118[0] : !llvm<"{ i32, i64 }">
    %120 = llvm.insertvalue %117, %119[1] : !llvm<"{ i32, i64 }">
    %121 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %122 = llvm.getelementptr %103[%121, %121] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %123 = llvm.load %122 : !llvm<"i32*">
    %124 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %125 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %126 = llvm.getelementptr %103[%124, %125] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %127 = llvm.load %126 : !llvm<"i64*">
    %128 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %129 = llvm.insertvalue %123, %128[0] : !llvm<"{ i32, i64 }">
    %130 = llvm.insertvalue %127, %129[1] : !llvm<"{ i32, i64 }">
    %131 = llvm.call @lua_add(%120, %130) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }">
    %132 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %133 = llvm.alloca %132 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %134 = llvm.extractvalue %131[0] : !llvm<"{ i32, i64 }">
    %135 = llvm.extractvalue %131[1] : !llvm<"{ i32, i64 }">
    %136 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %137 = llvm.getelementptr %133[%136, %136] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %134, %137 : !llvm<"i32*">
    %138 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %139 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %140 = llvm.getelementptr %133[%138, %139] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %135, %140 : !llvm<"i64*">
    %141 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %142 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %143 = llvm.getelementptr %10[%141, %142] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %144 = llvm.bitcast %143 : !llvm<"i64*"> to !llvm<"i8**">
    %145 = llvm.load %144 : !llvm<"i8**">
    %146 = llvm.call @lua_table_get_prealloc_impl(%145, %1) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"{ i32, i64 }">
    %147 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %148 = llvm.alloca %147 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %149 = llvm.extractvalue %146[0] : !llvm<"{ i32, i64 }">
    %150 = llvm.extractvalue %146[1] : !llvm<"{ i32, i64 }">
    %151 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %152 = llvm.getelementptr %148[%151, %151] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %149, %152 : !llvm<"i32*">
    %153 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %154 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %155 = llvm.getelementptr %148[%153, %154] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %150, %155 : !llvm<"i64*">
    %156 = llvm.mlir.addressof @g_arg_pack_ptr : !llvm<"i64*">
    %157 = llvm.bitcast %156 : !llvm<"i64*"> to !llvm<"i8**">
    %158 = llvm.load %157 : !llvm<"i8**">
    %159 = llvm.call @realloc(%158, %5) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    llvm.store %159, %157 : !llvm<"i8**">
    %160 = llvm.mlir.undef : !llvm<"{ i32, { i32, i64 }* }">
    %161 = llvm.bitcast %159 : !llvm<"i8*"> to !llvm<"{ i32, i64 }*">
    %162 = llvm.insertvalue %3, %160[0] : !llvm<"{ i32, { i32, i64 }* }">
    %163 = llvm.insertvalue %161, %162[1] : !llvm<"{ i32, { i32, i64 }* }">
    %164 = llvm.extractvalue %163[1] : !llvm<"{ i32, { i32, i64 }* }">
    %165 = llvm.getelementptr %164[%4] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %166 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %167 = llvm.getelementptr %148[%166, %166] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %168 = llvm.load %167 : !llvm<"i32*">
    %169 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %170 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %171 = llvm.getelementptr %148[%169, %170] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %172 = llvm.load %171 : !llvm<"i64*">
    %173 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %174 = llvm.insertvalue %168, %173[0] : !llvm<"{ i32, i64 }">
    %175 = llvm.insertvalue %172, %174[1] : !llvm<"{ i32, i64 }">
    llvm.store %175, %165 : !llvm<"{ i32, i64 }*">
    %176 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %177 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %178 = llvm.getelementptr %7[%176, %177] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %179 = llvm.bitcast %178 : !llvm<"i64*"> to !llvm<"i8**">
    %180 = llvm.load %179 : !llvm<"i8**">
    %181 = llvm.bitcast %180 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %182 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %183 = llvm.getelementptr %181[%182, %182] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %184 = llvm.load %183 : !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %185 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %186 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %187 = llvm.getelementptr %7[%185, %186] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %188 = llvm.bitcast %187 : !llvm<"i64*"> to !llvm<"i8**">
    %189 = llvm.load %188 : !llvm<"i8**">
    %190 = llvm.bitcast %189 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %191 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %192 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %193 = llvm.getelementptr %190[%191, %192] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, i64 }***">
    %194 = llvm.load %193 : !llvm<"{ i32, i64 }***">
    %195 = llvm.call %184(%194, %163) : (!llvm<"{ i32, i64 }**">, !llvm<"{ i32, { i32, i64 }* }">) -> !llvm<"{ i32, { i32, i64 }* }">
    %196 = llvm.extractvalue %195[1] : !llvm<"{ i32, { i32, i64 }* }">
    %197 = llvm.getelementptr %196[%4] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %198 = llvm.load %197 : !llvm<"{ i32, i64 }*">
    %199 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %200 = llvm.alloca %199 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %201 = llvm.extractvalue %198[0] : !llvm<"{ i32, i64 }">
    %202 = llvm.extractvalue %198[1] : !llvm<"{ i32, i64 }">
    %203 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %204 = llvm.getelementptr %200[%203, %203] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %201, %204 : !llvm<"i32*">
    %205 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %206 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %207 = llvm.getelementptr %200[%205, %206] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %202, %207 : !llvm<"i64*">
    %208 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %209 = llvm.getelementptr %133[%208, %208] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %210 = llvm.load %209 : !llvm<"i32*">
    %211 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %212 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %213 = llvm.getelementptr %133[%211, %212] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %214 = llvm.load %213 : !llvm<"i64*">
    %215 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %216 = llvm.insertvalue %210, %215[0] : !llvm<"{ i32, i64 }">
    %217 = llvm.insertvalue %214, %216[1] : !llvm<"{ i32, i64 }">
    %218 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %219 = llvm.getelementptr %200[%218, %218] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %220 = llvm.load %219 : !llvm<"i32*">
    %221 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %222 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %223 = llvm.getelementptr %200[%221, %222] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %224 = llvm.load %223 : !llvm<"i64*">
    %225 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %226 = llvm.insertvalue %220, %225[0] : !llvm<"{ i32, i64 }">
    %227 = llvm.insertvalue %224, %226[1] : !llvm<"{ i32, i64 }">
    %228 = llvm.call @lua_sub(%217, %227) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }">
    %229 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %230 = llvm.alloca %229 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %231 = llvm.extractvalue %228[0] : !llvm<"{ i32, i64 }">
    %232 = llvm.extractvalue %228[1] : !llvm<"{ i32, i64 }">
    %233 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %234 = llvm.getelementptr %230[%233, %233] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %231, %234 : !llvm<"i32*">
    %235 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %236 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %237 = llvm.getelementptr %230[%235, %236] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %232, %237 : !llvm<"i64*">
    %238 = llvm.mlir.addressof @g_ret_pack_ptr : !llvm<"i64*">
    %239 = llvm.bitcast %238 : !llvm<"i64*"> to !llvm<"i8**">
    %240 = llvm.load %239 : !llvm<"i8**">
    %241 = llvm.call @realloc(%240, %5) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    llvm.store %241, %239 : !llvm<"i8**">
    %242 = llvm.mlir.undef : !llvm<"{ i32, { i32, i64 }* }">
    %243 = llvm.bitcast %241 : !llvm<"i8*"> to !llvm<"{ i32, i64 }*">
    %244 = llvm.insertvalue %3, %242[0] : !llvm<"{ i32, { i32, i64 }* }">
    %245 = llvm.insertvalue %243, %244[1] : !llvm<"{ i32, { i32, i64 }* }">
    %246 = llvm.extractvalue %245[1] : !llvm<"{ i32, { i32, i64 }* }">
    %247 = llvm.getelementptr %246[%4] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %248 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %249 = llvm.getelementptr %230[%248, %248] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %250 = llvm.load %249 : !llvm<"i32*">
    %251 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %252 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %253 = llvm.getelementptr %230[%251, %252] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %254 = llvm.load %253 : !llvm<"i64*">
    %255 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %256 = llvm.insertvalue %250, %255[0] : !llvm<"{ i32, i64 }">
    %257 = llvm.insertvalue %254, %256[1] : !llvm<"{ i32, i64 }">
    llvm.store %257, %247 : !llvm<"{ i32, i64 }*">
    llvm.return %245 : !llvm<"{ i32, { i32, i64 }* }">
  ^bb2:  // pred: ^bb0
    %258 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %259 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %260 = llvm.getelementptr %10[%258, %259] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %261 = llvm.bitcast %260 : !llvm<"i64*"> to !llvm<"i8**">
    %262 = llvm.load %261 : !llvm<"i8**">
    %263 = llvm.call @lua_table_get_prealloc_impl(%262, %2) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"{ i32, i64 }">
    %264 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %265 = llvm.alloca %264 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %266 = llvm.extractvalue %263[0] : !llvm<"{ i32, i64 }">
    %267 = llvm.extractvalue %263[1] : !llvm<"{ i32, i64 }">
    %268 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %269 = llvm.getelementptr %265[%268, %268] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %266, %269 : !llvm<"i32*">
    %270 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %271 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %272 = llvm.getelementptr %265[%270, %271] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %267, %272 : !llvm<"i64*">
    %273 = llvm.mlir.addressof @g_ret_pack_ptr : !llvm<"i64*">
    %274 = llvm.bitcast %273 : !llvm<"i64*"> to !llvm<"i8**">
    %275 = llvm.load %274 : !llvm<"i8**">
    %276 = llvm.call @realloc(%275, %5) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    llvm.store %276, %274 : !llvm<"i8**">
    %277 = llvm.mlir.undef : !llvm<"{ i32, { i32, i64 }* }">
    %278 = llvm.bitcast %276 : !llvm<"i8*"> to !llvm<"{ i32, i64 }*">
    %279 = llvm.insertvalue %3, %277[0] : !llvm<"{ i32, { i32, i64 }* }">
    %280 = llvm.insertvalue %278, %279[1] : !llvm<"{ i32, { i32, i64 }* }">
    %281 = llvm.extractvalue %280[1] : !llvm<"{ i32, { i32, i64 }* }">
    %282 = llvm.getelementptr %281[%4] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %283 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %284 = llvm.getelementptr %265[%283, %283] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %285 = llvm.load %284 : !llvm<"i32*">
    %286 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %287 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %288 = llvm.getelementptr %265[%286, %287] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %289 = llvm.load %288 : !llvm<"i64*">
    %290 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %291 = llvm.insertvalue %285, %290[0] : !llvm<"{ i32, i64 }">
    %292 = llvm.insertvalue %289, %291[1] : !llvm<"{ i32, i64 }">
    llvm.store %292, %282 : !llvm<"{ i32, i64 }*">
    llvm.return %280 : !llvm<"{ i32, { i32, i64 }* }">
  }
  llvm.func @lua_anon_fcn_1(%arg0: !llvm<"{ i32, i64 }**">, %arg1: !llvm<"{ i32, { i32, i64 }* }">) -> !llvm<"{ i32, { i32, i64 }* }"> {
    %0 = llvm.mlir.constant(1 : i64) : !llvm.i64
    %1 = llvm.mlir.constant(2 : i64) : !llvm.i64
    %2 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %3 = llvm.mlir.constant(0.000000e+00 : f64) : !llvm.double
    %4 = llvm.mlir.constant(1.000000e+00 : f64) : !llvm.double
    %5 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %6 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %7 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %8 = llvm.mlir.constant(2 : index) : !llvm.i64
    %9 = llvm.mlir.constant(1 : index) : !llvm.i64
    %10 = llvm.getelementptr %arg0[%7] : (!llvm<"{ i32, i64 }**">, !llvm.i32) -> !llvm<"{ i32, i64 }**">
    %11 = llvm.load %10 : !llvm<"{ i32, i64 }**">
    %12 = llvm.call @lua_pack_get(%arg1, %7) : (!llvm<"{ i32, { i32, i64 }* }">, !llvm.i32) -> !llvm<"{ i32, i64 }">
    %13 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %14 = llvm.alloca %13 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %15 = llvm.extractvalue %12[0] : !llvm<"{ i32, i64 }">
    %16 = llvm.extractvalue %12[1] : !llvm<"{ i32, i64 }">
    %17 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %18 = llvm.getelementptr %14[%17, %17] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %15, %18 : !llvm<"i32*">
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %21 = llvm.getelementptr %14[%19, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %16, %21 : !llvm<"i64*">
    %22 = llvm.call @lua_pack_get(%arg1, %6) : (!llvm<"{ i32, { i32, i64 }* }">, !llvm.i32) -> !llvm<"{ i32, i64 }">
    %23 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %24 = llvm.alloca %23 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %25 = llvm.extractvalue %22[0] : !llvm<"{ i32, i64 }">
    %26 = llvm.extractvalue %22[1] : !llvm<"{ i32, i64 }">
    %27 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %28 = llvm.getelementptr %24[%27, %27] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %25, %28 : !llvm<"i32*">
    %29 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.getelementptr %24[%29, %30] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %26, %31 : !llvm<"i64*">
    %32 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %33 = llvm.alloca %32 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %34 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %35 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %36 = llvm.getelementptr %33[%35, %35] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %34, %36 : !llvm<"i32*">
    %37 = llvm.bitcast %3 : !llvm.double to !llvm.i64
    %38 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %39 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %40 = llvm.getelementptr %33[%38, %39] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %37, %40 : !llvm<"i64*">
    %41 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %42 = llvm.getelementptr %24[%41, %41] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %43 = llvm.load %42 : !llvm<"i32*">
    %44 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %45 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %46 = llvm.getelementptr %24[%44, %45] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %47 = llvm.load %46 : !llvm<"i64*">
    %48 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %49 = llvm.insertvalue %43, %48[0] : !llvm<"{ i32, i64 }">
    %50 = llvm.insertvalue %47, %49[1] : !llvm<"{ i32, i64 }">
    %51 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %52 = llvm.getelementptr %33[%51, %51] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %53 = llvm.load %52 : !llvm<"i32*">
    %54 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %55 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %56 = llvm.getelementptr %33[%54, %55] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %57 = llvm.load %56 : !llvm<"i64*">
    %58 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %59 = llvm.insertvalue %53, %58[0] : !llvm<"{ i32, i64 }">
    %60 = llvm.insertvalue %57, %59[1] : !llvm<"{ i32, i64 }">
    %61 = llvm.call @lua_gt(%50, %60) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }">
    %62 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %63 = llvm.alloca %62 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %64 = llvm.extractvalue %61[0] : !llvm<"{ i32, i64 }">
    %65 = llvm.extractvalue %61[1] : !llvm<"{ i32, i64 }">
    %66 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %67 = llvm.getelementptr %63[%66, %66] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %64, %67 : !llvm<"i32*">
    %68 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %69 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %70 = llvm.getelementptr %63[%68, %69] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %65, %70 : !llvm<"i64*">
    %71 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %72 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %73 = llvm.getelementptr %63[%71, %72] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %74 = llvm.load %73 : !llvm<"i64*">
    %75 = llvm.trunc %74 : !llvm.i64 to !llvm.i1
    llvm.cond_br %75, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %76 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %77 = llvm.getelementptr %14[%76, %76] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %78 = llvm.load %77 : !llvm<"i32*">
    %79 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %80 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %81 = llvm.getelementptr %14[%79, %80] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %82 = llvm.load %81 : !llvm<"i64*">
    %83 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %84 = llvm.insertvalue %78, %83[0] : !llvm<"{ i32, i64 }">
    %85 = llvm.insertvalue %82, %84[1] : !llvm<"{ i32, i64 }">
    %86 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %87 = llvm.getelementptr %14[%86, %86] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %88 = llvm.load %87 : !llvm<"i32*">
    %89 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %90 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %91 = llvm.getelementptr %14[%89, %90] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %92 = llvm.load %91 : !llvm<"i64*">
    %93 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %94 = llvm.insertvalue %88, %93[0] : !llvm<"{ i32, i64 }">
    %95 = llvm.insertvalue %92, %94[1] : !llvm<"{ i32, i64 }">
    %96 = llvm.call @lua_add(%85, %95) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }">
    %97 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %98 = llvm.alloca %97 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %99 = llvm.extractvalue %96[0] : !llvm<"{ i32, i64 }">
    %100 = llvm.extractvalue %96[1] : !llvm<"{ i32, i64 }">
    %101 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %102 = llvm.getelementptr %98[%101, %101] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %99, %102 : !llvm<"i32*">
    %103 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %104 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %105 = llvm.getelementptr %98[%103, %104] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %100, %105 : !llvm<"i64*">
    %106 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %107 = llvm.alloca %106 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %108 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %109 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %110 = llvm.getelementptr %107[%109, %109] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %108, %110 : !llvm<"i32*">
    %111 = llvm.bitcast %4 : !llvm.double to !llvm.i64
    %112 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %113 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %114 = llvm.getelementptr %107[%112, %113] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %111, %114 : !llvm<"i64*">
    %115 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %116 = llvm.getelementptr %24[%115, %115] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %117 = llvm.load %116 : !llvm<"i32*">
    %118 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %119 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %120 = llvm.getelementptr %24[%118, %119] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %121 = llvm.load %120 : !llvm<"i64*">
    %122 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %123 = llvm.insertvalue %117, %122[0] : !llvm<"{ i32, i64 }">
    %124 = llvm.insertvalue %121, %123[1] : !llvm<"{ i32, i64 }">
    %125 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %126 = llvm.getelementptr %107[%125, %125] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %127 = llvm.load %126 : !llvm<"i32*">
    %128 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %129 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %130 = llvm.getelementptr %107[%128, %129] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %131 = llvm.load %130 : !llvm<"i64*">
    %132 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %133 = llvm.insertvalue %127, %132[0] : !llvm<"{ i32, i64 }">
    %134 = llvm.insertvalue %131, %133[1] : !llvm<"{ i32, i64 }">
    %135 = llvm.call @lua_sub(%124, %134) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }">
    %136 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %137 = llvm.alloca %136 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %138 = llvm.extractvalue %135[0] : !llvm<"{ i32, i64 }">
    %139 = llvm.extractvalue %135[1] : !llvm<"{ i32, i64 }">
    %140 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %141 = llvm.getelementptr %137[%140, %140] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %138, %141 : !llvm<"i32*">
    %142 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %143 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %144 = llvm.getelementptr %137[%142, %143] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %139, %144 : !llvm<"i64*">
    %145 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %146 = llvm.getelementptr %137[%145, %145] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %147 = llvm.load %146 : !llvm<"i32*">
    %148 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %149 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %150 = llvm.getelementptr %137[%148, %149] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %151 = llvm.load %150 : !llvm<"i64*">
    %152 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %153 = llvm.getelementptr %24[%152, %152] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %147, %153 : !llvm<"i32*">
    %154 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %155 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %156 = llvm.getelementptr %24[%154, %155] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %151, %156 : !llvm<"i64*">
    %157 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %158 = llvm.getelementptr %98[%157, %157] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %159 = llvm.load %158 : !llvm<"i32*">
    %160 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %161 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %162 = llvm.getelementptr %98[%160, %161] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %163 = llvm.load %162 : !llvm<"i64*">
    %164 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %165 = llvm.insertvalue %159, %164[0] : !llvm<"{ i32, i64 }">
    %166 = llvm.insertvalue %163, %165[1] : !llvm<"{ i32, i64 }">
    %167 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %168 = llvm.getelementptr %107[%167, %167] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %169 = llvm.load %168 : !llvm<"i32*">
    %170 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %171 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %172 = llvm.getelementptr %107[%170, %171] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %173 = llvm.load %172 : !llvm<"i64*">
    %174 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %175 = llvm.insertvalue %169, %174[0] : !llvm<"{ i32, i64 }">
    %176 = llvm.insertvalue %173, %175[1] : !llvm<"{ i32, i64 }">
    %177 = llvm.call @lua_sub(%166, %176) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }">
    %178 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %179 = llvm.alloca %178 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %180 = llvm.extractvalue %177[0] : !llvm<"{ i32, i64 }">
    %181 = llvm.extractvalue %177[1] : !llvm<"{ i32, i64 }">
    %182 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %183 = llvm.getelementptr %179[%182, %182] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %180, %183 : !llvm<"i32*">
    %184 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %185 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %186 = llvm.getelementptr %179[%184, %185] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %181, %186 : !llvm<"i64*">
    %187 = llvm.mlir.addressof @g_arg_pack_ptr : !llvm<"i64*">
    %188 = llvm.bitcast %187 : !llvm<"i64*"> to !llvm<"i8**">
    %189 = llvm.load %188 : !llvm<"i8**">
    %190 = llvm.call @realloc(%189, %8) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    llvm.store %190, %188 : !llvm<"i8**">
    %191 = llvm.mlir.undef : !llvm<"{ i32, { i32, i64 }* }">
    %192 = llvm.bitcast %190 : !llvm<"i8*"> to !llvm<"{ i32, i64 }*">
    %193 = llvm.insertvalue %5, %191[0] : !llvm<"{ i32, { i32, i64 }* }">
    %194 = llvm.insertvalue %192, %193[1] : !llvm<"{ i32, { i32, i64 }* }">
    %195 = llvm.extractvalue %194[1] : !llvm<"{ i32, { i32, i64 }* }">
    %196 = llvm.getelementptr %195[%7] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %197 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %198 = llvm.getelementptr %179[%197, %197] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %199 = llvm.load %198 : !llvm<"i32*">
    %200 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %201 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %202 = llvm.getelementptr %179[%200, %201] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %203 = llvm.load %202 : !llvm<"i64*">
    %204 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %205 = llvm.insertvalue %199, %204[0] : !llvm<"{ i32, i64 }">
    %206 = llvm.insertvalue %203, %205[1] : !llvm<"{ i32, i64 }">
    llvm.store %206, %196 : !llvm<"{ i32, i64 }*">
    %207 = llvm.extractvalue %194[1] : !llvm<"{ i32, { i32, i64 }* }">
    %208 = llvm.getelementptr %207[%6] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %209 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %210 = llvm.getelementptr %24[%209, %209] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %211 = llvm.load %210 : !llvm<"i32*">
    %212 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %213 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %214 = llvm.getelementptr %24[%212, %213] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %215 = llvm.load %214 : !llvm<"i64*">
    %216 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %217 = llvm.insertvalue %211, %216[0] : !llvm<"{ i32, i64 }">
    %218 = llvm.insertvalue %215, %217[1] : !llvm<"{ i32, i64 }">
    llvm.store %218, %208 : !llvm<"{ i32, i64 }*">
    %219 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %220 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %221 = llvm.getelementptr %11[%219, %220] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %222 = llvm.bitcast %221 : !llvm<"i64*"> to !llvm<"i8**">
    %223 = llvm.load %222 : !llvm<"i8**">
    %224 = llvm.bitcast %223 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %225 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %226 = llvm.getelementptr %224[%225, %225] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %227 = llvm.load %226 : !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %228 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %229 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %230 = llvm.getelementptr %11[%228, %229] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %231 = llvm.bitcast %230 : !llvm<"i64*"> to !llvm<"i8**">
    %232 = llvm.load %231 : !llvm<"i8**">
    %233 = llvm.bitcast %232 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %234 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %235 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %236 = llvm.getelementptr %233[%234, %235] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, i64 }***">
    %237 = llvm.load %236 : !llvm<"{ i32, i64 }***">
    %238 = llvm.call %227(%237, %194) : (!llvm<"{ i32, i64 }**">, !llvm<"{ i32, { i32, i64 }* }">) -> !llvm<"{ i32, { i32, i64 }* }">
    %239 = llvm.extractvalue %238[1] : !llvm<"{ i32, { i32, i64 }* }">
    %240 = llvm.getelementptr %239[%7] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %241 = llvm.load %240 : !llvm<"{ i32, i64 }*">
    %242 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %243 = llvm.alloca %242 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %244 = llvm.extractvalue %241[0] : !llvm<"{ i32, i64 }">
    %245 = llvm.extractvalue %241[1] : !llvm<"{ i32, i64 }">
    %246 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %247 = llvm.getelementptr %243[%246, %246] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %244, %247 : !llvm<"i32*">
    %248 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %249 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %250 = llvm.getelementptr %243[%248, %249] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %245, %250 : !llvm<"i64*">
    %251 = llvm.mlir.addressof @g_arg_pack_ptr : !llvm<"i64*">
    %252 = llvm.bitcast %251 : !llvm<"i64*"> to !llvm<"i8**">
    %253 = llvm.load %252 : !llvm<"i8**">
    %254 = llvm.call @realloc(%253, %8) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    llvm.store %254, %252 : !llvm<"i8**">
    %255 = llvm.mlir.undef : !llvm<"{ i32, { i32, i64 }* }">
    %256 = llvm.bitcast %254 : !llvm<"i8*"> to !llvm<"{ i32, i64 }*">
    %257 = llvm.insertvalue %5, %255[0] : !llvm<"{ i32, { i32, i64 }* }">
    %258 = llvm.insertvalue %256, %257[1] : !llvm<"{ i32, { i32, i64 }* }">
    %259 = llvm.extractvalue %258[1] : !llvm<"{ i32, { i32, i64 }* }">
    %260 = llvm.getelementptr %259[%7] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %261 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %262 = llvm.getelementptr %98[%261, %261] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %263 = llvm.load %262 : !llvm<"i32*">
    %264 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %265 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %266 = llvm.getelementptr %98[%264, %265] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %267 = llvm.load %266 : !llvm<"i64*">
    %268 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %269 = llvm.insertvalue %263, %268[0] : !llvm<"{ i32, i64 }">
    %270 = llvm.insertvalue %267, %269[1] : !llvm<"{ i32, i64 }">
    llvm.store %270, %260 : !llvm<"{ i32, i64 }*">
    %271 = llvm.extractvalue %258[1] : !llvm<"{ i32, { i32, i64 }* }">
    %272 = llvm.getelementptr %271[%6] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %273 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %274 = llvm.getelementptr %24[%273, %273] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %275 = llvm.load %274 : !llvm<"i32*">
    %276 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %277 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %278 = llvm.getelementptr %24[%276, %277] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %279 = llvm.load %278 : !llvm<"i64*">
    %280 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %281 = llvm.insertvalue %275, %280[0] : !llvm<"{ i32, i64 }">
    %282 = llvm.insertvalue %279, %281[1] : !llvm<"{ i32, i64 }">
    llvm.store %282, %272 : !llvm<"{ i32, i64 }*">
    %283 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %284 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %285 = llvm.getelementptr %11[%283, %284] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %286 = llvm.bitcast %285 : !llvm<"i64*"> to !llvm<"i8**">
    %287 = llvm.load %286 : !llvm<"i8**">
    %288 = llvm.bitcast %287 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %289 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %290 = llvm.getelementptr %288[%289, %289] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %291 = llvm.load %290 : !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">
    %292 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %293 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %294 = llvm.getelementptr %11[%292, %293] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %295 = llvm.bitcast %294 : !llvm<"i64*"> to !llvm<"i8**">
    %296 = llvm.load %295 : !llvm<"i8**">
    %297 = llvm.bitcast %296 : !llvm<"i8*"> to !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
    %298 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %299 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %300 = llvm.getelementptr %297[%298, %299] : (!llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i32, i64 }***">
    %301 = llvm.load %300 : !llvm<"{ i32, i64 }***">
    %302 = llvm.call %291(%301, %258) : (!llvm<"{ i32, i64 }**">, !llvm<"{ i32, { i32, i64 }* }">) -> !llvm<"{ i32, { i32, i64 }* }">
    %303 = llvm.extractvalue %302[1] : !llvm<"{ i32, { i32, i64 }* }">
    %304 = llvm.getelementptr %303[%7] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %305 = llvm.load %304 : !llvm<"{ i32, i64 }*">
    %306 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %307 = llvm.alloca %306 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %308 = llvm.extractvalue %305[0] : !llvm<"{ i32, i64 }">
    %309 = llvm.extractvalue %305[1] : !llvm<"{ i32, i64 }">
    %310 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %311 = llvm.getelementptr %307[%310, %310] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %308, %311 : !llvm<"i32*">
    %312 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %313 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %314 = llvm.getelementptr %307[%312, %313] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %309, %314 : !llvm<"i64*">
    %315 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %316 = llvm.alloca %315 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %317 = llvm.mlir.constant(4 : i32) : !llvm.i32
    %318 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %319 = llvm.getelementptr %316[%318, %318] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %317, %319 : !llvm<"i32*">
    %320 = llvm.call @lua_new_table_impl() : () -> !llvm<"i8*">
    %321 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %322 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %323 = llvm.getelementptr %316[%321, %322] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %324 = llvm.bitcast %323 : !llvm<"i64*"> to !llvm<"i8**">
    llvm.store %320, %324 : !llvm<"i8**">
    %325 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %326 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %327 = llvm.getelementptr %316[%325, %326] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %328 = llvm.bitcast %327 : !llvm<"i64*"> to !llvm<"i8**">
    %329 = llvm.load %328 : !llvm<"i8**">
    %330 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %331 = llvm.getelementptr %14[%330, %330] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %332 = llvm.load %331 : !llvm<"i32*">
    %333 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %334 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %335 = llvm.getelementptr %14[%333, %334] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %336 = llvm.load %335 : !llvm<"i64*">
    llvm.call @lua_table_set_prealloc_impl(%329, %2, %332, %336) : (!llvm<"i8*">, !llvm.i64, !llvm.i32, !llvm.i64) -> ()
    %337 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %338 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %339 = llvm.getelementptr %316[%337, %338] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %340 = llvm.bitcast %339 : !llvm<"i64*"> to !llvm<"i8**">
    %341 = llvm.load %340 : !llvm<"i8**">
    %342 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %343 = llvm.getelementptr %243[%342, %342] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %344 = llvm.load %343 : !llvm<"i32*">
    %345 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %346 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %347 = llvm.getelementptr %243[%345, %346] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %348 = llvm.load %347 : !llvm<"i64*">
    llvm.call @lua_table_set_prealloc_impl(%341, %0, %344, %348) : (!llvm<"i8*">, !llvm.i64, !llvm.i32, !llvm.i64) -> ()
    %349 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %350 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %351 = llvm.getelementptr %316[%349, %350] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %352 = llvm.bitcast %351 : !llvm<"i64*"> to !llvm<"i8**">
    %353 = llvm.load %352 : !llvm<"i8**">
    %354 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %355 = llvm.getelementptr %307[%354, %354] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %356 = llvm.load %355 : !llvm<"i32*">
    %357 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %358 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %359 = llvm.getelementptr %307[%357, %358] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %360 = llvm.load %359 : !llvm<"i64*">
    llvm.call @lua_table_set_prealloc_impl(%353, %1, %356, %360) : (!llvm<"i8*">, !llvm.i64, !llvm.i32, !llvm.i64) -> ()
    %361 = llvm.mlir.addressof @g_ret_pack_ptr : !llvm<"i64*">
    %362 = llvm.bitcast %361 : !llvm<"i64*"> to !llvm<"i8**">
    %363 = llvm.load %362 : !llvm<"i8**">
    %364 = llvm.call @realloc(%363, %9) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    llvm.store %364, %362 : !llvm<"i8**">
    %365 = llvm.mlir.undef : !llvm<"{ i32, { i32, i64 }* }">
    %366 = llvm.bitcast %364 : !llvm<"i8*"> to !llvm<"{ i32, i64 }*">
    %367 = llvm.insertvalue %6, %365[0] : !llvm<"{ i32, { i32, i64 }* }">
    %368 = llvm.insertvalue %366, %367[1] : !llvm<"{ i32, { i32, i64 }* }">
    %369 = llvm.extractvalue %368[1] : !llvm<"{ i32, { i32, i64 }* }">
    %370 = llvm.getelementptr %369[%7] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %371 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %372 = llvm.getelementptr %316[%371, %371] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %373 = llvm.load %372 : !llvm<"i32*">
    %374 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %375 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %376 = llvm.getelementptr %316[%374, %375] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %377 = llvm.load %376 : !llvm<"i64*">
    %378 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %379 = llvm.insertvalue %373, %378[0] : !llvm<"{ i32, i64 }">
    %380 = llvm.insertvalue %377, %379[1] : !llvm<"{ i32, i64 }">
    llvm.store %380, %370 : !llvm<"{ i32, i64 }*">
    llvm.return %368 : !llvm<"{ i32, { i32, i64 }* }">
  ^bb2:  // pred: ^bb0
    %381 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %382 = llvm.alloca %381 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %383 = llvm.mlir.constant(4 : i32) : !llvm.i32
    %384 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %385 = llvm.getelementptr %382[%384, %384] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %383, %385 : !llvm<"i32*">
    %386 = llvm.call @lua_new_table_impl() : () -> !llvm<"i8*">
    %387 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %388 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %389 = llvm.getelementptr %382[%387, %388] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %390 = llvm.bitcast %389 : !llvm<"i64*"> to !llvm<"i8**">
    llvm.store %386, %390 : !llvm<"i8**">
    %391 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %392 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %393 = llvm.getelementptr %382[%391, %392] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %394 = llvm.bitcast %393 : !llvm<"i64*"> to !llvm<"i8**">
    %395 = llvm.load %394 : !llvm<"i8**">
    %396 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %397 = llvm.getelementptr %14[%396, %396] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %398 = llvm.load %397 : !llvm<"i32*">
    %399 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %400 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %401 = llvm.getelementptr %14[%399, %400] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %402 = llvm.load %401 : !llvm<"i64*">
    llvm.call @lua_table_set_prealloc_impl(%395, %2, %398, %402) : (!llvm<"i8*">, !llvm.i64, !llvm.i32, !llvm.i64) -> ()
    %403 = llvm.mlir.addressof @g_ret_pack_ptr : !llvm<"i64*">
    %404 = llvm.bitcast %403 : !llvm<"i64*"> to !llvm<"i8**">
    %405 = llvm.load %404 : !llvm<"i8**">
    %406 = llvm.call @realloc(%405, %9) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
    llvm.store %406, %404 : !llvm<"i8**">
    %407 = llvm.mlir.undef : !llvm<"{ i32, { i32, i64 }* }">
    %408 = llvm.bitcast %406 : !llvm<"i8*"> to !llvm<"{ i32, i64 }*">
    %409 = llvm.insertvalue %6, %407[0] : !llvm<"{ i32, { i32, i64 }* }">
    %410 = llvm.insertvalue %408, %409[1] : !llvm<"{ i32, { i32, i64 }* }">
    %411 = llvm.extractvalue %410[1] : !llvm<"{ i32, { i32, i64 }* }">
    %412 = llvm.getelementptr %411[%7] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %413 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %414 = llvm.getelementptr %382[%413, %413] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %415 = llvm.load %414 : !llvm<"i32*">
    %416 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %417 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %418 = llvm.getelementptr %382[%416, %417] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %419 = llvm.load %418 : !llvm<"i64*">
    %420 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %421 = llvm.insertvalue %415, %420[0] : !llvm<"{ i32, i64 }">
    %422 = llvm.insertvalue %419, %421[1] : !llvm<"{ i32, i64 }">
    llvm.store %422, %412 : !llvm<"{ i32, i64 }*">
    llvm.return %410 : !llvm<"{ i32, { i32, i64 }* }">
  }
  llvm.func @print_one(!llvm<"{ i32, i64 }">)
  llvm.func @luac_check_number_type(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm.i1 {
    %0 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %2 = llvm.alloca %1 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %3 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %4 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %5 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %6 = llvm.getelementptr %2[%5, %5] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %3, %6 : !llvm<"i32*">
    %7 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %8 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %9 = llvm.getelementptr %2[%7, %8] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %4, %9 : !llvm<"i64*">
    %10 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %11 = llvm.alloca %10 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %12 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %13 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %14 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %15 = llvm.getelementptr %11[%14, %14] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %12, %15 : !llvm<"i32*">
    %16 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %17 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %18 = llvm.getelementptr %11[%16, %17] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %13, %18 : !llvm<"i64*">
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.getelementptr %2[%19, %19] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %21 = llvm.load %20 : !llvm<"i32*">
    %22 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %23 = llvm.getelementptr %11[%22, %22] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %24 = llvm.load %23 : !llvm<"i32*">
    %25 = llvm.icmp "eq" %21, %0 : !llvm.i32
    %26 = llvm.icmp "eq" %24, %0 : !llvm.i32
    %27 = llvm.and %25, %26 : !llvm.i1
    llvm.return %27 : !llvm.i1
  }
  llvm.func @lua_add(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1 = llvm.alloca %0 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %2 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %3 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %5 = llvm.getelementptr %1[%4, %4] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %2, %5 : !llvm<"i32*">
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %8 = llvm.getelementptr %1[%6, %7] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %3, %8 : !llvm<"i64*">
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.alloca %9 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %11 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %12 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %13 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %14 = llvm.getelementptr %10[%13, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %11, %14 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %10[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %12, %17 : !llvm<"i64*">
    %18 = llvm.call @luac_check_number_type(%arg0, %arg1) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm.i1
    llvm.cond_br %18, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %21 = llvm.getelementptr %1[%19, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %22 = llvm.load %21 : !llvm<"i64*">
    %23 = llvm.bitcast %22 : !llvm.i64 to !llvm.double
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %26 = llvm.getelementptr %10[%24, %25] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %27 = llvm.load %26 : !llvm<"i64*">
    %28 = llvm.bitcast %27 : !llvm.i64 to !llvm.double
    %29 = llvm.fadd %23, %28 : !llvm.double
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.alloca %30 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %32 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.getelementptr %31[%33, %33] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %32, %34 : !llvm<"i32*">
    %35 = llvm.bitcast %29 : !llvm.double to !llvm.i64
    %36 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %37 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %38 = llvm.getelementptr %31[%36, %37] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %35, %38 : !llvm<"i64*">
    %39 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %40 = llvm.getelementptr %31[%39, %39] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %41 = llvm.load %40 : !llvm<"i32*">
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %44 = llvm.getelementptr %31[%42, %43] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %45 = llvm.load %44 : !llvm<"i64*">
    %46 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %47 = llvm.insertvalue %41, %46[0] : !llvm<"{ i32, i64 }">
    %48 = llvm.insertvalue %45, %47[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%48 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %49 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %50 = llvm.alloca %49 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %51 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %53 = llvm.getelementptr %50[%52, %52] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %51, %53 : !llvm<"i32*">
    %54 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %55 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %56 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %57 = llvm.getelementptr %50[%55, %56] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %54, %57 : !llvm<"i64*">
    %58 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %59 = llvm.getelementptr %50[%58, %58] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %60 = llvm.load %59 : !llvm<"i32*">
    %61 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %62 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %63 = llvm.getelementptr %50[%61, %62] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %64 = llvm.load %63 : !llvm<"i64*">
    %65 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %66 = llvm.insertvalue %60, %65[0] : !llvm<"{ i32, i64 }">
    %67 = llvm.insertvalue %64, %66[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%67 : !llvm<"{ i32, i64 }">)
  ^bb3(%68: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %68 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_sub(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1 = llvm.alloca %0 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %2 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %3 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %5 = llvm.getelementptr %1[%4, %4] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %2, %5 : !llvm<"i32*">
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %8 = llvm.getelementptr %1[%6, %7] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %3, %8 : !llvm<"i64*">
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.alloca %9 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %11 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %12 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %13 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %14 = llvm.getelementptr %10[%13, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %11, %14 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %10[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %12, %17 : !llvm<"i64*">
    %18 = llvm.call @luac_check_number_type(%arg0, %arg1) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm.i1
    llvm.cond_br %18, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %21 = llvm.getelementptr %1[%19, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %22 = llvm.load %21 : !llvm<"i64*">
    %23 = llvm.bitcast %22 : !llvm.i64 to !llvm.double
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %26 = llvm.getelementptr %10[%24, %25] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %27 = llvm.load %26 : !llvm<"i64*">
    %28 = llvm.bitcast %27 : !llvm.i64 to !llvm.double
    %29 = llvm.fsub %23, %28 : !llvm.double
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.alloca %30 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %32 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.getelementptr %31[%33, %33] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %32, %34 : !llvm<"i32*">
    %35 = llvm.bitcast %29 : !llvm.double to !llvm.i64
    %36 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %37 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %38 = llvm.getelementptr %31[%36, %37] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %35, %38 : !llvm<"i64*">
    %39 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %40 = llvm.getelementptr %31[%39, %39] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %41 = llvm.load %40 : !llvm<"i32*">
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %44 = llvm.getelementptr %31[%42, %43] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %45 = llvm.load %44 : !llvm<"i64*">
    %46 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %47 = llvm.insertvalue %41, %46[0] : !llvm<"{ i32, i64 }">
    %48 = llvm.insertvalue %45, %47[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%48 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %49 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %50 = llvm.alloca %49 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %51 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %53 = llvm.getelementptr %50[%52, %52] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %51, %53 : !llvm<"i32*">
    %54 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %55 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %56 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %57 = llvm.getelementptr %50[%55, %56] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %54, %57 : !llvm<"i64*">
    %58 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %59 = llvm.getelementptr %50[%58, %58] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %60 = llvm.load %59 : !llvm<"i32*">
    %61 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %62 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %63 = llvm.getelementptr %50[%61, %62] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %64 = llvm.load %63 : !llvm<"i64*">
    %65 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %66 = llvm.insertvalue %60, %65[0] : !llvm<"{ i32, i64 }">
    %67 = llvm.insertvalue %64, %66[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%67 : !llvm<"{ i32, i64 }">)
  ^bb3(%68: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %68 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_mul(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1 = llvm.alloca %0 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %2 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %3 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %5 = llvm.getelementptr %1[%4, %4] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %2, %5 : !llvm<"i32*">
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %8 = llvm.getelementptr %1[%6, %7] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %3, %8 : !llvm<"i64*">
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.alloca %9 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %11 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %12 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %13 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %14 = llvm.getelementptr %10[%13, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %11, %14 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %10[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %12, %17 : !llvm<"i64*">
    %18 = llvm.call @luac_check_number_type(%arg0, %arg1) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm.i1
    llvm.cond_br %18, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %21 = llvm.getelementptr %1[%19, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %22 = llvm.load %21 : !llvm<"i64*">
    %23 = llvm.bitcast %22 : !llvm.i64 to !llvm.double
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %26 = llvm.getelementptr %10[%24, %25] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %27 = llvm.load %26 : !llvm<"i64*">
    %28 = llvm.bitcast %27 : !llvm.i64 to !llvm.double
    %29 = llvm.fmul %23, %28 : !llvm.double
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.alloca %30 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %32 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.getelementptr %31[%33, %33] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %32, %34 : !llvm<"i32*">
    %35 = llvm.bitcast %29 : !llvm.double to !llvm.i64
    %36 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %37 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %38 = llvm.getelementptr %31[%36, %37] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %35, %38 : !llvm<"i64*">
    %39 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %40 = llvm.getelementptr %31[%39, %39] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %41 = llvm.load %40 : !llvm<"i32*">
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %44 = llvm.getelementptr %31[%42, %43] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %45 = llvm.load %44 : !llvm<"i64*">
    %46 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %47 = llvm.insertvalue %41, %46[0] : !llvm<"{ i32, i64 }">
    %48 = llvm.insertvalue %45, %47[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%48 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %49 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %50 = llvm.alloca %49 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %51 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %53 = llvm.getelementptr %50[%52, %52] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %51, %53 : !llvm<"i32*">
    %54 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %55 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %56 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %57 = llvm.getelementptr %50[%55, %56] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %54, %57 : !llvm<"i64*">
    %58 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %59 = llvm.getelementptr %50[%58, %58] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %60 = llvm.load %59 : !llvm<"i32*">
    %61 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %62 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %63 = llvm.getelementptr %50[%61, %62] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %64 = llvm.load %63 : !llvm<"i64*">
    %65 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %66 = llvm.insertvalue %60, %65[0] : !llvm<"{ i32, i64 }">
    %67 = llvm.insertvalue %64, %66[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%67 : !llvm<"{ i32, i64 }">)
  ^bb3(%68: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %68 : !llvm<"{ i32, i64 }">
  }
  llvm.func @pow(!llvm.double, !llvm.double) -> !llvm.double
  llvm.func @lua_pow(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1 = llvm.alloca %0 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %2 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %3 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %5 = llvm.getelementptr %1[%4, %4] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %2, %5 : !llvm<"i32*">
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %8 = llvm.getelementptr %1[%6, %7] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %3, %8 : !llvm<"i64*">
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.alloca %9 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %11 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %12 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %13 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %14 = llvm.getelementptr %10[%13, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %11, %14 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %10[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %12, %17 : !llvm<"i64*">
    %18 = llvm.call @luac_check_number_type(%arg0, %arg1) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm.i1
    llvm.cond_br %18, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %21 = llvm.getelementptr %1[%19, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %22 = llvm.load %21 : !llvm<"i64*">
    %23 = llvm.bitcast %22 : !llvm.i64 to !llvm.double
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %26 = llvm.getelementptr %10[%24, %25] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %27 = llvm.load %26 : !llvm<"i64*">
    %28 = llvm.bitcast %27 : !llvm.i64 to !llvm.double
    %29 = llvm.call @pow(%23, %28) : (!llvm.double, !llvm.double) -> !llvm.double
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.alloca %30 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %32 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.getelementptr %31[%33, %33] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %32, %34 : !llvm<"i32*">
    %35 = llvm.bitcast %29 : !llvm.double to !llvm.i64
    %36 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %37 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %38 = llvm.getelementptr %31[%36, %37] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %35, %38 : !llvm<"i64*">
    %39 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %40 = llvm.getelementptr %31[%39, %39] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %41 = llvm.load %40 : !llvm<"i32*">
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %44 = llvm.getelementptr %31[%42, %43] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %45 = llvm.load %44 : !llvm<"i64*">
    %46 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %47 = llvm.insertvalue %41, %46[0] : !llvm<"{ i32, i64 }">
    %48 = llvm.insertvalue %45, %47[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%48 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %49 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %50 = llvm.alloca %49 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %51 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %53 = llvm.getelementptr %50[%52, %52] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %51, %53 : !llvm<"i32*">
    %54 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %55 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %56 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %57 = llvm.getelementptr %50[%55, %56] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %54, %57 : !llvm<"i64*">
    %58 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %59 = llvm.getelementptr %50[%58, %58] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %60 = llvm.load %59 : !llvm<"i32*">
    %61 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %62 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %63 = llvm.getelementptr %50[%61, %62] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %64 = llvm.load %63 : !llvm<"i64*">
    %65 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %66 = llvm.insertvalue %60, %65[0] : !llvm<"{ i32, i64 }">
    %67 = llvm.insertvalue %64, %66[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%67 : !llvm<"{ i32, i64 }">)
  ^bb3(%68: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %68 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_neg(%arg0: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(-1.000000e+00 : f64) : !llvm.double
    %2 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %3 = llvm.alloca %2 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %4 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %5 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.getelementptr %3[%6, %6] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %4, %7 : !llvm<"i32*">
    %8 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.getelementptr %3[%8, %9] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %5, %10 : !llvm<"i64*">
    %11 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %12 = llvm.getelementptr %3[%11, %11] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %13 = llvm.load %12 : !llvm<"i32*">
    %14 = llvm.icmp "eq" %0, %13 : !llvm.i32
    llvm.cond_br %14, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %3[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %18 = llvm.load %17 : !llvm<"i64*">
    %19 = llvm.bitcast %18 : !llvm.i64 to !llvm.double
    %20 = llvm.fmul %1, %19 : !llvm.double
    %21 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %22 = llvm.alloca %21 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %23 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.getelementptr %22[%24, %24] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %23, %25 : !llvm<"i32*">
    %26 = llvm.bitcast %20 : !llvm.double to !llvm.i64
    %27 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %28 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %29 = llvm.getelementptr %22[%27, %28] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %26, %29 : !llvm<"i64*">
    %30 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %31 = llvm.getelementptr %22[%30, %30] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %32 = llvm.load %31 : !llvm<"i32*">
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %35 = llvm.getelementptr %22[%33, %34] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %36 = llvm.load %35 : !llvm<"i64*">
    %37 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %38 = llvm.insertvalue %32, %37[0] : !llvm<"{ i32, i64 }">
    %39 = llvm.insertvalue %36, %38[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%39 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %40 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %41 = llvm.alloca %40 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %44 = llvm.getelementptr %41[%43, %43] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %42, %44 : !llvm<"i32*">
    %45 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %46 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %47 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %48 = llvm.getelementptr %41[%46, %47] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %45, %48 : !llvm<"i64*">
    %49 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %50 = llvm.getelementptr %41[%49, %49] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %51 = llvm.load %50 : !llvm<"i32*">
    %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %53 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %54 = llvm.getelementptr %41[%52, %53] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %55 = llvm.load %54 : !llvm<"i64*">
    %56 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %57 = llvm.insertvalue %51, %56[0] : !llvm<"{ i32, i64 }">
    %58 = llvm.insertvalue %55, %57[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%58 : !llvm<"{ i32, i64 }">)
  ^bb3(%59: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %59 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_lt(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1 = llvm.alloca %0 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %2 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %3 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %5 = llvm.getelementptr %1[%4, %4] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %2, %5 : !llvm<"i32*">
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %8 = llvm.getelementptr %1[%6, %7] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %3, %8 : !llvm<"i64*">
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.alloca %9 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %11 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %12 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %13 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %14 = llvm.getelementptr %10[%13, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %11, %14 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %10[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %12, %17 : !llvm<"i64*">
    %18 = llvm.call @luac_check_number_type(%arg0, %arg1) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm.i1
    llvm.cond_br %18, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %21 = llvm.getelementptr %1[%19, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %22 = llvm.load %21 : !llvm<"i64*">
    %23 = llvm.bitcast %22 : !llvm.i64 to !llvm.double
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %26 = llvm.getelementptr %10[%24, %25] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %27 = llvm.load %26 : !llvm<"i64*">
    %28 = llvm.bitcast %27 : !llvm.i64 to !llvm.double
    %29 = llvm.fcmp "olt" %23, %28 : !llvm.double
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.alloca %30 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %32 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.getelementptr %31[%33, %33] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %32, %34 : !llvm<"i32*">
    %35 = llvm.zext %29 : !llvm.i1 to !llvm.i64
    %36 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %37 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %38 = llvm.getelementptr %31[%36, %37] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %35, %38 : !llvm<"i64*">
    %39 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %40 = llvm.getelementptr %31[%39, %39] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %41 = llvm.load %40 : !llvm<"i32*">
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %44 = llvm.getelementptr %31[%42, %43] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %45 = llvm.load %44 : !llvm<"i64*">
    %46 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %47 = llvm.insertvalue %41, %46[0] : !llvm<"{ i32, i64 }">
    %48 = llvm.insertvalue %45, %47[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%48 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %49 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %50 = llvm.alloca %49 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %51 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %53 = llvm.getelementptr %50[%52, %52] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %51, %53 : !llvm<"i32*">
    %54 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %55 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %56 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %57 = llvm.getelementptr %50[%55, %56] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %54, %57 : !llvm<"i64*">
    %58 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %59 = llvm.getelementptr %50[%58, %58] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %60 = llvm.load %59 : !llvm<"i32*">
    %61 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %62 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %63 = llvm.getelementptr %50[%61, %62] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %64 = llvm.load %63 : !llvm<"i64*">
    %65 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %66 = llvm.insertvalue %60, %65[0] : !llvm<"{ i32, i64 }">
    %67 = llvm.insertvalue %64, %66[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%67 : !llvm<"{ i32, i64 }">)
  ^bb3(%68: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %68 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_gt(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1 = llvm.alloca %0 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %2 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %3 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %5 = llvm.getelementptr %1[%4, %4] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %2, %5 : !llvm<"i32*">
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %8 = llvm.getelementptr %1[%6, %7] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %3, %8 : !llvm<"i64*">
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.alloca %9 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %11 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %12 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %13 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %14 = llvm.getelementptr %10[%13, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %11, %14 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %10[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %12, %17 : !llvm<"i64*">
    %18 = llvm.call @luac_check_number_type(%arg0, %arg1) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm.i1
    llvm.cond_br %18, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %21 = llvm.getelementptr %1[%19, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %22 = llvm.load %21 : !llvm<"i64*">
    %23 = llvm.bitcast %22 : !llvm.i64 to !llvm.double
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %26 = llvm.getelementptr %10[%24, %25] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %27 = llvm.load %26 : !llvm<"i64*">
    %28 = llvm.bitcast %27 : !llvm.i64 to !llvm.double
    %29 = llvm.fcmp "ogt" %23, %28 : !llvm.double
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.alloca %30 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %32 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.getelementptr %31[%33, %33] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %32, %34 : !llvm<"i32*">
    %35 = llvm.zext %29 : !llvm.i1 to !llvm.i64
    %36 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %37 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %38 = llvm.getelementptr %31[%36, %37] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %35, %38 : !llvm<"i64*">
    %39 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %40 = llvm.getelementptr %31[%39, %39] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %41 = llvm.load %40 : !llvm<"i32*">
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %44 = llvm.getelementptr %31[%42, %43] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %45 = llvm.load %44 : !llvm<"i64*">
    %46 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %47 = llvm.insertvalue %41, %46[0] : !llvm<"{ i32, i64 }">
    %48 = llvm.insertvalue %45, %47[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%48 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %49 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %50 = llvm.alloca %49 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %51 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %53 = llvm.getelementptr %50[%52, %52] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %51, %53 : !llvm<"i32*">
    %54 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %55 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %56 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %57 = llvm.getelementptr %50[%55, %56] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %54, %57 : !llvm<"i64*">
    %58 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %59 = llvm.getelementptr %50[%58, %58] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %60 = llvm.load %59 : !llvm<"i32*">
    %61 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %62 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %63 = llvm.getelementptr %50[%61, %62] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %64 = llvm.load %63 : !llvm<"i64*">
    %65 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %66 = llvm.insertvalue %60, %65[0] : !llvm<"{ i32, i64 }">
    %67 = llvm.insertvalue %64, %66[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%67 : !llvm<"{ i32, i64 }">)
  ^bb3(%68: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %68 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_le(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1 = llvm.alloca %0 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %2 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %3 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %5 = llvm.getelementptr %1[%4, %4] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %2, %5 : !llvm<"i32*">
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %8 = llvm.getelementptr %1[%6, %7] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %3, %8 : !llvm<"i64*">
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.alloca %9 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %11 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %12 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %13 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %14 = llvm.getelementptr %10[%13, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %11, %14 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %10[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %12, %17 : !llvm<"i64*">
    %18 = llvm.call @luac_check_number_type(%arg0, %arg1) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm.i1
    llvm.cond_br %18, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %21 = llvm.getelementptr %1[%19, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %22 = llvm.load %21 : !llvm<"i64*">
    %23 = llvm.bitcast %22 : !llvm.i64 to !llvm.double
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %26 = llvm.getelementptr %10[%24, %25] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %27 = llvm.load %26 : !llvm<"i64*">
    %28 = llvm.bitcast %27 : !llvm.i64 to !llvm.double
    %29 = llvm.fcmp "ole" %23, %28 : !llvm.double
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.alloca %30 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %32 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.getelementptr %31[%33, %33] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %32, %34 : !llvm<"i32*">
    %35 = llvm.zext %29 : !llvm.i1 to !llvm.i64
    %36 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %37 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %38 = llvm.getelementptr %31[%36, %37] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %35, %38 : !llvm<"i64*">
    %39 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %40 = llvm.getelementptr %31[%39, %39] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %41 = llvm.load %40 : !llvm<"i32*">
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %44 = llvm.getelementptr %31[%42, %43] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %45 = llvm.load %44 : !llvm<"i64*">
    %46 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %47 = llvm.insertvalue %41, %46[0] : !llvm<"{ i32, i64 }">
    %48 = llvm.insertvalue %45, %47[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%48 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %49 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %50 = llvm.alloca %49 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %51 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %53 = llvm.getelementptr %50[%52, %52] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %51, %53 : !llvm<"i32*">
    %54 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %55 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %56 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %57 = llvm.getelementptr %50[%55, %56] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %54, %57 : !llvm<"i64*">
    %58 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %59 = llvm.getelementptr %50[%58, %58] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %60 = llvm.load %59 : !llvm<"i32*">
    %61 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %62 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %63 = llvm.getelementptr %50[%61, %62] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %64 = llvm.load %63 : !llvm<"i64*">
    %65 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %66 = llvm.insertvalue %60, %65[0] : !llvm<"{ i32, i64 }">
    %67 = llvm.insertvalue %64, %66[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%67 : !llvm<"{ i32, i64 }">)
  ^bb3(%68: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %68 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_bool_and(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.call @lua_convert_bool_like(%arg0) : (!llvm<"{ i32, i64 }">) -> !llvm.i1
    %1 = llvm.call @lua_convert_bool_like(%arg1) : (!llvm<"{ i32, i64 }">) -> !llvm.i1
    %2 = llvm.and %0, %1 : !llvm.i1
    %3 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %4 = llvm.alloca %3 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %5 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.getelementptr %4[%6, %6] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %5, %7 : !llvm<"i32*">
    %8 = llvm.zext %2 : !llvm.i1 to !llvm.i64
    %9 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %10 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %11 = llvm.getelementptr %4[%9, %10] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %8, %11 : !llvm<"i64*">
    %12 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %13 = llvm.getelementptr %4[%12, %12] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %14 = llvm.load %13 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %4[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %18 = llvm.load %17 : !llvm<"i64*">
    %19 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %20 = llvm.insertvalue %14, %19[0] : !llvm<"{ i32, i64 }">
    %21 = llvm.insertvalue %18, %20[1] : !llvm<"{ i32, i64 }">
    llvm.return %21 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_bool_not(%arg0: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(true) : !llvm.i1
    %1 = llvm.call @lua_convert_bool_like(%arg0) : (!llvm<"{ i32, i64 }">) -> !llvm.i1
    %2 = llvm.xor %1, %0 : !llvm.i1
    %3 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %4 = llvm.alloca %3 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %5 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.getelementptr %4[%6, %6] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %5, %7 : !llvm<"i32*">
    %8 = llvm.zext %2 : !llvm.i1 to !llvm.i64
    %9 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %10 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %11 = llvm.getelementptr %4[%9, %10] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %8, %11 : !llvm<"i64*">
    %12 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %13 = llvm.getelementptr %4[%12, %12] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %14 = llvm.load %13 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %4[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %18 = llvm.load %17 : !llvm<"i64*">
    %19 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %20 = llvm.insertvalue %14, %19[0] : !llvm<"{ i32, i64 }">
    %21 = llvm.insertvalue %18, %20[1] : !llvm<"{ i32, i64 }">
    llvm.return %21 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_list_size_impl(!llvm<"i8*">) -> !llvm.i64
  llvm.func @lua_list_size(%arg0: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(4 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %2 = llvm.alloca %1 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %3 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %4 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %5 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %6 = llvm.getelementptr %2[%5, %5] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %3, %6 : !llvm<"i32*">
    %7 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %8 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %9 = llvm.getelementptr %2[%7, %8] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %4, %9 : !llvm<"i64*">
    %10 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %11 = llvm.getelementptr %2[%10, %10] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %12 = llvm.load %11 : !llvm<"i32*">
    %13 = llvm.icmp "eq" %12, %0 : !llvm.i32
    llvm.cond_br %13, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %14 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %15 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %16 = llvm.getelementptr %2[%14, %15] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %17 = llvm.bitcast %16 : !llvm<"i64*"> to !llvm<"i8**">
    %18 = llvm.load %17 : !llvm<"i8**">
    %19 = llvm.call @lua_list_size_impl(%18) : (!llvm<"i8*">) -> !llvm.i64
    %20 = llvm.sitofp %19 : !llvm.i64 to !llvm.double
    %21 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %22 = llvm.alloca %21 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %23 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.getelementptr %22[%24, %24] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %23, %25 : !llvm<"i32*">
    %26 = llvm.bitcast %20 : !llvm.double to !llvm.i64
    %27 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %28 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %29 = llvm.getelementptr %22[%27, %28] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %26, %29 : !llvm<"i64*">
    %30 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %31 = llvm.getelementptr %22[%30, %30] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %32 = llvm.load %31 : !llvm<"i32*">
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %35 = llvm.getelementptr %22[%33, %34] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %36 = llvm.load %35 : !llvm<"i64*">
    %37 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %38 = llvm.insertvalue %32, %37[0] : !llvm<"{ i32, i64 }">
    %39 = llvm.insertvalue %36, %38[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%39 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %40 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %41 = llvm.alloca %40 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %44 = llvm.getelementptr %41[%43, %43] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %42, %44 : !llvm<"i32*">
    %45 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %46 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %47 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %48 = llvm.getelementptr %41[%46, %47] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %45, %48 : !llvm<"i64*">
    %49 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %50 = llvm.getelementptr %41[%49, %49] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %51 = llvm.load %50 : !llvm<"i32*">
    %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %53 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %54 = llvm.getelementptr %41[%52, %53] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %55 = llvm.load %54 : !llvm<"i64*">
    %56 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %57 = llvm.insertvalue %51, %56[0] : !llvm<"{ i32, i64 }">
    %58 = llvm.insertvalue %55, %57[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%58 : !llvm<"{ i32, i64 }">)
  ^bb3(%59: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %59 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_strcat_impl(!llvm<"i8*">, !llvm<"i8*">) -> !llvm<"{ i32, i64 }">
  llvm.func @lua_strcat(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(3 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %2 = llvm.alloca %1 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %3 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %4 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %5 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %6 = llvm.getelementptr %2[%5, %5] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %3, %6 : !llvm<"i32*">
    %7 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %8 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %9 = llvm.getelementptr %2[%7, %8] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %4, %9 : !llvm<"i64*">
    %10 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %11 = llvm.alloca %10 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %12 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %13 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %14 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %15 = llvm.getelementptr %11[%14, %14] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %12, %15 : !llvm<"i32*">
    %16 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %17 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %18 = llvm.getelementptr %11[%16, %17] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %13, %18 : !llvm<"i64*">
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.getelementptr %2[%19, %19] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %21 = llvm.load %20 : !llvm<"i32*">
    %22 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %23 = llvm.getelementptr %11[%22, %22] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %24 = llvm.load %23 : !llvm<"i32*">
    %25 = llvm.icmp "eq" %21, %0 : !llvm.i32
    %26 = llvm.icmp "eq" %24, %0 : !llvm.i32
    %27 = llvm.and %25, %26 : !llvm.i1
    llvm.cond_br %27, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %28 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %29 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %30 = llvm.getelementptr %2[%28, %29] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %31 = llvm.bitcast %30 : !llvm<"i64*"> to !llvm<"i8**">
    %32 = llvm.load %31 : !llvm<"i8**">
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %35 = llvm.getelementptr %11[%33, %34] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %36 = llvm.bitcast %35 : !llvm<"i64*"> to !llvm<"i8**">
    %37 = llvm.load %36 : !llvm<"i8**">
    %38 = llvm.call @lua_strcat_impl(%32, %37) : (!llvm<"i8*">, !llvm<"i8*">) -> !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%38 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %39 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %40 = llvm.alloca %39 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %41 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.getelementptr %40[%42, %42] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %41, %43 : !llvm<"i32*">
    %44 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %45 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %46 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %47 = llvm.getelementptr %40[%45, %46] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %44, %47 : !llvm<"i64*">
    %48 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %49 = llvm.getelementptr %40[%48, %48] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %50 = llvm.load %49 : !llvm<"i32*">
    %51 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %52 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %53 = llvm.getelementptr %40[%51, %52] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %54 = llvm.load %53 : !llvm<"i64*">
    %55 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %56 = llvm.insertvalue %50, %55[0] : !llvm<"{ i32, i64 }">
    %57 = llvm.insertvalue %54, %56[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%57 : !llvm<"{ i32, i64 }">)
  ^bb3(%58: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %58 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_eq_impl(!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm.i1
  llvm.func @lua_eq(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(false) : !llvm.i1
    %1 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %2 = llvm.alloca %1 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %3 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %4 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %5 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %6 = llvm.getelementptr %2[%5, %5] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %3, %6 : !llvm<"i32*">
    %7 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %8 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %9 = llvm.getelementptr %2[%7, %8] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %4, %9 : !llvm<"i64*">
    %10 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %11 = llvm.alloca %10 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %12 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %13 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %14 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %15 = llvm.getelementptr %11[%14, %14] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %12, %15 : !llvm<"i32*">
    %16 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %17 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %18 = llvm.getelementptr %11[%16, %17] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %13, %18 : !llvm<"i64*">
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.getelementptr %2[%19, %19] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %21 = llvm.load %20 : !llvm<"i32*">
    %22 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %23 = llvm.getelementptr %11[%22, %22] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %24 = llvm.load %23 : !llvm<"i32*">
    %25 = llvm.icmp "eq" %21, %24 : !llvm.i32
    llvm.cond_br %25, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %26 = llvm.call @lua_eq_impl(%arg0, %arg1) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm.i1
    llvm.br ^bb3(%26 : !llvm.i1)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%0 : !llvm.i1)
  ^bb3(%27: !llvm.i1):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    %28 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %29 = llvm.alloca %28 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %32 = llvm.getelementptr %29[%31, %31] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %30, %32 : !llvm<"i32*">
    %33 = llvm.zext %27 : !llvm.i1 to !llvm.i64
    %34 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %35 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %36 = llvm.getelementptr %29[%34, %35] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %33, %36 : !llvm<"i64*">
    %37 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %38 = llvm.getelementptr %29[%37, %37] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %39 = llvm.load %38 : !llvm<"i32*">
    %40 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %41 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %42 = llvm.getelementptr %29[%40, %41] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %43 = llvm.load %42 : !llvm<"i64*">
    %44 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %45 = llvm.insertvalue %39, %44[0] : !llvm<"{ i32, i64 }">
    %46 = llvm.insertvalue %43, %45[1] : !llvm<"{ i32, i64 }">
    llvm.return %46 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_ne(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(true) : !llvm.i1
    %1 = llvm.call @lua_eq(%arg0, %arg1) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }">
    %2 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %3 = llvm.alloca %2 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %4 = llvm.extractvalue %1[0] : !llvm<"{ i32, i64 }">
    %5 = llvm.extractvalue %1[1] : !llvm<"{ i32, i64 }">
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.getelementptr %3[%6, %6] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %4, %7 : !llvm<"i32*">
    %8 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.getelementptr %3[%8, %9] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %5, %10 : !llvm<"i64*">
    %11 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %12 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %13 = llvm.getelementptr %3[%11, %12] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %14 = llvm.load %13 : !llvm<"i64*">
    %15 = llvm.trunc %14 : !llvm.i64 to !llvm.i1
    %16 = llvm.xor %15, %0 : !llvm.i1
    %17 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %18 = llvm.alloca %17 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %19 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %21 = llvm.getelementptr %18[%20, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %19, %21 : !llvm<"i32*">
    %22 = llvm.zext %16 : !llvm.i1 to !llvm.i64
    %23 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %24 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %25 = llvm.getelementptr %18[%23, %24] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %22, %25 : !llvm<"i64*">
    %26 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %27 = llvm.getelementptr %18[%26, %26] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %28 = llvm.load %27 : !llvm<"i32*">
    %29 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.getelementptr %18[%29, %30] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %32 = llvm.load %31 : !llvm<"i64*">
    %33 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %34 = llvm.insertvalue %28, %33[0] : !llvm<"{ i32, i64 }">
    %35 = llvm.insertvalue %32, %34[1] : !llvm<"{ i32, i64 }">
    llvm.return %35 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_convert_bool_like(%arg0: !llvm<"{ i32, i64 }">) -> !llvm.i1 {
    %0 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(false) : !llvm.i1
    %2 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %3 = llvm.mlir.constant(true) : !llvm.i1
    %4 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %5 = llvm.alloca %4 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %6 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %7 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %8 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %9 = llvm.getelementptr %5[%8, %8] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %6, %9 : !llvm<"i32*">
    %10 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %11 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %12 = llvm.getelementptr %5[%10, %11] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %7, %12 : !llvm<"i64*">
    %13 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %14 = llvm.getelementptr %5[%13, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %15 = llvm.load %14 : !llvm<"i32*">
    %16 = llvm.icmp "eq" %15, %0 : !llvm.i32
    llvm.cond_br %16, ^bb1(%1 : !llvm.i1), ^bb2
  ^bb1(%17: !llvm.i1):  // 2 preds: ^bb0, ^bb5
    llvm.br ^bb6(%17 : !llvm.i1)
  ^bb2:  // pred: ^bb0
    %18 = llvm.icmp "eq" %15, %2 : !llvm.i32
    llvm.cond_br %18, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %21 = llvm.getelementptr %5[%19, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %22 = llvm.load %21 : !llvm<"i64*">
    %23 = llvm.trunc %22 : !llvm.i64 to !llvm.i1
    llvm.br ^bb5(%23 : !llvm.i1)
  ^bb4:  // pred: ^bb2
    llvm.br ^bb5(%3 : !llvm.i1)
  ^bb5(%24: !llvm.i1):  // 2 preds: ^bb3, ^bb4
    llvm.br ^bb1(%24 : !llvm.i1)
  ^bb6(%25: !llvm.i1):  // pred: ^bb1
    llvm.br ^bb7
  ^bb7:  // pred: ^bb6
    llvm.return %25 : !llvm.i1
  }
  llvm.func @lua_pack_insert_all(%arg0: !llvm<"{ i32, { i32, i64 }* }">, %arg1: !llvm<"{ i32, { i32, i64 }* }">, %arg2: !llvm.i32) {
    %0 = llvm.mlir.constant(0 : index) : !llvm.i64
    %1 = llvm.mlir.constant(1 : index) : !llvm.i64
    %2 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, { i32, i64 }* }">
    %3 = llvm.sext %2 : !llvm.i32 to !llvm.i64
    llvm.br ^bb1(%0 : !llvm.i64)
  ^bb1(%4: !llvm.i64):  // 2 preds: ^bb0, ^bb2
    %5 = llvm.icmp "slt" %4, %3 : !llvm.i64
    llvm.cond_br %5, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %6 = llvm.trunc %4 : !llvm.i64 to !llvm.i32
    %7 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, { i32, i64 }* }">
    %8 = llvm.getelementptr %7[%6] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %9 = llvm.load %8 : !llvm<"{ i32, i64 }*">
    %10 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %11 = llvm.alloca %10 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %12 = llvm.extractvalue %9[0] : !llvm<"{ i32, i64 }">
    %13 = llvm.extractvalue %9[1] : !llvm<"{ i32, i64 }">
    %14 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %15 = llvm.getelementptr %11[%14, %14] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %12, %15 : !llvm<"i32*">
    %16 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %17 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %18 = llvm.getelementptr %11[%16, %17] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %13, %18 : !llvm<"i64*">
    %19 = llvm.add %arg2, %6 : !llvm.i32
    %20 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, { i32, i64 }* }">
    %21 = llvm.getelementptr %20[%19] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %22 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %23 = llvm.getelementptr %11[%22, %22] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %24 = llvm.load %23 : !llvm<"i32*">
    %25 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %26 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %27 = llvm.getelementptr %11[%25, %26] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %28 = llvm.load %27 : !llvm<"i64*">
    %29 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %30 = llvm.insertvalue %24, %29[0] : !llvm<"{ i32, i64 }">
    %31 = llvm.insertvalue %28, %30[1] : !llvm<"{ i32, i64 }">
    llvm.store %31, %21 : !llvm<"{ i32, i64 }*">
    %32 = llvm.add %4, %1 : !llvm.i64
    llvm.br ^bb1(%32 : !llvm.i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @lua_pack_get(%arg0: !llvm<"{ i32, { i32, i64 }* }">, %arg1: !llvm.i32) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, { i32, i64 }* }">
    %1 = llvm.icmp "slt" %arg1, %0 : !llvm.i32
    llvm.cond_br %1, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %2 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, { i32, i64 }* }">
    %3 = llvm.getelementptr %2[%arg1] : (!llvm<"{ i32, i64 }*">, !llvm.i32) -> !llvm<"{ i32, i64 }*">
    %4 = llvm.load %3 : !llvm<"{ i32, i64 }*">
    %5 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %6 = llvm.alloca %5 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %7 = llvm.extractvalue %4[0] : !llvm<"{ i32, i64 }">
    %8 = llvm.extractvalue %4[1] : !llvm<"{ i32, i64 }">
    %9 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %10 = llvm.getelementptr %6[%9, %9] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %7, %10 : !llvm<"i32*">
    %11 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %12 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %13 = llvm.getelementptr %6[%11, %12] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %8, %13 : !llvm<"i64*">
    %14 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %15 = llvm.getelementptr %6[%14, %14] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %16 = llvm.load %15 : !llvm<"i32*">
    %17 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %18 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %19 = llvm.getelementptr %6[%17, %18] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %20 = llvm.load %19 : !llvm<"i64*">
    %21 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %22 = llvm.insertvalue %16, %21[0] : !llvm<"{ i32, i64 }">
    %23 = llvm.insertvalue %20, %22[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%23 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %24 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %25 = llvm.alloca %24 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %26 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %27 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %28 = llvm.getelementptr %25[%27, %27] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %26, %28 : !llvm<"i32*">
    %29 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %30 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %31 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %32 = llvm.getelementptr %25[%30, %31] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %29, %32 : !llvm<"i64*">
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.getelementptr %25[%33, %33] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %35 = llvm.load %34 : !llvm<"i32*">
    %36 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %37 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %38 = llvm.getelementptr %25[%36, %37] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %39 = llvm.load %38 : !llvm<"i64*">
    %40 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %41 = llvm.insertvalue %35, %40[0] : !llvm<"{ i32, i64 }">
    %42 = llvm.insertvalue %39, %41[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%42 : !llvm<"{ i32, i64 }">)
  ^bb3(%43: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %43 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_table_get_impl(!llvm<"i8*">, !llvm.i32, !llvm.i64) -> !llvm<"{ i32, i64 }">
  llvm.func @lua_table_set_impl(!llvm<"i8*">, !llvm.i32, !llvm.i64, !llvm.i32, !llvm.i64)
  llvm.func @lua_table_get_prealloc_impl(!llvm<"i8*">, !llvm.i64) -> !llvm<"{ i32, i64 }">
  llvm.func @lua_table_set_prealloc_impl(!llvm<"i8*">, !llvm.i64, !llvm.i32, !llvm.i64)
  llvm.func @lua_make_fcn_impl(!llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*">, !llvm<"{ i32, i64 }**">) -> !llvm<"i8*">
  llvm.func @lua_load_string_impl(!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
  llvm.func @lua_new_table_impl() -> !llvm<"i8*">
  llvm.mlir.global internal @lua_anon_string_0("stretch tree of depth")
  llvm.mlir.global internal @lua_anon_string_1("check:")
  llvm.mlir.global internal @lua_anon_string_2("trees of depth")
  llvm.mlir.global internal @lua_anon_string_3("long lived tree of depth")
  llvm.func @malloc(!llvm.i64) -> !llvm<"i8*">
  llvm.func @free(!llvm<"i8*">)
  llvm.func @realloc(!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
  llvm.mlir.global internal @g_arg_pack_ptr(0 : i64) : !llvm.i64
  llvm.mlir.global internal @g_ret_pack_ptr(0 : i64) : !llvm.i64
  llvm.mlir.global external @lua_builtin_math() : !llvm<"{ i32, i64 }">
  llvm.mlir.global external @lua_builtin_string() : !llvm<"{ i32, i64 }">
  llvm.mlir.global external @lua_builtin_table() : !llvm<"{ i32, i64 }">
  llvm.mlir.global external @lua_builtin_io() : !llvm<"{ i32, i64 }">
  llvm.mlir.global external @lua_builtin_print() : !llvm<"{ i32, i64 }">
}
