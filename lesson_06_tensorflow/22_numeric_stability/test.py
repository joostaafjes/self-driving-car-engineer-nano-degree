a = 1000000000
for i in range(1000000):
    a = a + 1e-6
print(a - 1000000000)

b = 0
for i in range(1000000):
    b = b + 1e-6
print(b)
