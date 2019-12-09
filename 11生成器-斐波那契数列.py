def fib(max):
    n = 0
    a = 0
    b = 1
    while n < max:
        yield(b)
        a,b = b,a+b
        n = n + 1
# 测试
f = fib(10)
for n in f:
    print(n)