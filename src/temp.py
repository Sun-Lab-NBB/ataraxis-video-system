def inner(x):
    print(x+4)

def outer(func, args=()):
    for i in range(4):
        func()

outer(lambda: inner(4))