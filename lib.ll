; ModuleID = 'lua/lib.c'
source_filename = "lua/lib.c"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

%struct.TObject = type { i32, %union.Value }
%union.Value = type { %struct.LuaNumber }
%struct.LuaNumber = type { %union.anon, i32 }
%union.anon = type { i64 }

; Function Attrs: noinline nounwind optnone ssp uwtable
define void @hold() #0 {
  %1 = alloca %struct.TObject, align 8
  %2 = alloca %struct.TObject, align 8
  %3 = alloca %struct.TObject, align 8
  %4 = alloca %struct.TObject, align 8
  call void @lua_add(%struct.TObject* sret %1, %struct.TObject* null, %struct.TObject* null)
  call void @lua_sub(%struct.TObject* sret %2, %struct.TObject* null, %struct.TObject* null)
  call void @lua_wrap_int(%struct.TObject* sret %3, i64 0)
  call void @lua_wrap_float(%struct.TObject* sret %4, double 0.000000e+00)
  ret void
}

declare void @lua_add(%struct.TObject* sret, %struct.TObject*, %struct.TObject*) #1

declare void @lua_sub(%struct.TObject* sret, %struct.TObject*, %struct.TObject*) #1

declare void @lua_wrap_int(%struct.TObject* sret, i64) #1

declare void @lua_wrap_float(%struct.TObject* sret, double) #1

attributes #0 = { noinline nounwind optnone ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang version 10.0.0 "}
