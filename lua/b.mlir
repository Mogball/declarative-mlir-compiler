%0 = get "a"
%n = constant 1
%1 = assign %0 = %n

%2 = get "a"
print(%2)

%3 = get "a"
%n = constant 2
%4 = assign %3 = %n

%x = get "a"
print(%x)

loop.for %i=1,10 {
  %l0 = get "a"
  %l1 = assign %l0 = %i
}

%5 = get "a"
print(%5)


//////////////////////////////////

%a = constant 1

print(%a)

%b = constant 2
%4 = assign %1 = %b

print(%b)

loop.for %i=1,10 {
  %l1 = assign %b = %i
}

print(%b)

