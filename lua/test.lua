function iter (a, i)
  i = i + 1
  local v = a[i]
  print(i,v)
  if i then
    print("xd")
  end
  if v then
    print("xddd")
  end
  if v then
    return i, v
  end
end

function ipairs (a)
  return iter, a, 0
end

tbl = {1, 2, 4}
for i,v in ipairs(tbl) do
  print(i,v)
end
