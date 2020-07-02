a = 0
b = 0
for i=5,10000,2 do
  a = (a + 5) * 1

  for j=1,10000,3 do
    b = b + i + j
  end

end

print(a)
print(b)
