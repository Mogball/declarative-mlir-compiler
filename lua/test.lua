--
-- lua.function_def @something [] () {
--   ....
--   lua.ret
-- }
--
--
--
a = 5
function something()
  print(42, a)
end

something()

function something()
  print(5, a)
end

something()
