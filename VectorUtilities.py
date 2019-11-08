def vector_add(x, y):
    """
    Vector x + Vector y
    """
    if (len(x) != len(y)):
        print("add Vectors not same len")
        print("x:")
        for i in range(len(x)):
            print(x[i])
        print("y:")
        for i in range(len(y)):
            print(y[i])
        return
        return

    z = []
    for i in range(len(x)):
        z.append(x[i]+y[i])

    return z

def vector_subtract (x, y):
    """
    Vector x - Vector y
    """
    z = []
    if (len(x) != len(y)):
        print("sub Vectors not same len")
        print("x:")
        for i in range(len(x)):
            print(x[i])
        print("y:")
        for i in range(len(y)):
            print(y[i])
        return

    for i in range(len(x)):
        z.append(x[i] - y[i])

    return z

def vector_magnitude_squared(x):
    """
    ||Vector x||^2
    """
    mag = 0
    for i in range(len(x)):
        mag += x[i] * x[i]

    return mag
