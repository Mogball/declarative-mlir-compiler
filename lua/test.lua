--
-- lua.function_def @something [] () {
--   ....
--   lua.ret
-- }
--
--
--
function do_print(a)
  print(a)
end

do_print(5)
do_print()
do_print(4, 6)
do_print(print)
