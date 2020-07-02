a = 5
for i=1,1 do
  print(a)
  local a
  print(a)
  a = 10
  print(a)

  for j=1,1 do
    a = 11
    print(a)
    local a = 99
    print(a)
  end


  print(a)


end

print(a)
