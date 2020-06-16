; ModuleID = 'lua/lib.c'
source_filename = "lua/lib.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.TObject = type { i32, %union.Value }
%union.Value = type { %struct.LuaNumber }
%struct.LuaNumber = type { %union.anon, i32 }
%union.anon = type { i64 }

; Function Attrs: noinline nounwind optnone sspstrong uwtable
define dso_local void @hold() #0 {
  ret void
}

declare void @lua_add(%struct.TObject* sret, %struct.TObject*, %struct.TObject*) #1

declare void @lua_sub(%struct.TObject* sret, %struct.TObject*, %struct.TObject*) #1

declare void @lua_eq(%struct.TObject* sret, %struct.TObject*, %struct.TObject*) #1

declare void @lua_neq(%struct.TObject* sret, %struct.TObject*, %struct.TObject*) #1

declare void @lua_get_nil(%struct.TObject* sret) #1

declare void @lua_new_table(%struct.TObject* sret) #1

declare void @lua_get_string(%struct.TObject* sret, i8*, i32) #1

declare void @lua_wrap_int(%struct.TObject* sret, i64) #1

declare void @lua_wrap_real(%struct.TObject* sret, double) #1

declare void @lua_wrap_bool(%struct.TObject* sret, i1 zeroext) #1

declare i64 @lua_unwrap_int(%struct.TObject*) #1

declare double @lua_unwrap_real(%struct.TObject*) #1

declare zeroext i1 @lua_unwrap_bool(%struct.TObject*) #1

declare void @lua_typeof(%struct.TObject* sret, %struct.TObject*) #1

declare void @lua_table_get(%struct.TObject* sret, %struct.TObject*, %struct.TObject*) #1

declare void @lua_table_set(%struct.TObject*, %struct.TObject*, %struct.TObject*) #1

declare void @lua_table_size(%struct.TObject* sret, %struct.TObject*) #1

attributes #0 = { noinline nounwind optnone sspstrong uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{!"clang version 10.0.0 "}
