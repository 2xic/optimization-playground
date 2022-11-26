from cudaplayground import tensor, pare_array


def test_subtract():
    a = tensor((2, 2)).ones()

    b = a - 2
    assert (b.isEqual(tensor((2, 2)).ones() * (-1)))

    c = 2 - a
    assert (c.isEqual(tensor((2, 2)).ones()))


def test_addition():
    a = tensor((2, 2)).zeros()

    b = a + 2
    assert (b.isEqual(tensor((2, 2)).ones() * (2)))

    c = 2 + a
    assert (c.isEqual(tensor((2, 2)).ones() * (2)))


def test_mul():
    a = tensor((2, 2)).ones()

    b = -1 * (a * -1)
    assert (b.isEqual(tensor((2, 2)).ones()))

    c = (a * -1) * -1 * 2
    assert (c.isEqual(tensor((2, 2)).ones() * 2))


def test_divide():
    a = tensor((2, 2)).ones() * 2

    b = 4 / a
    assert (b.isEqual(tensor((2, 2)).ones() * 2))

    b = a / 4
    assert (b.isEqual(tensor((2, 2)).ones() / 2))

    assert not (b.isEqual(tensor((2, 2)).ones()))


def test_negate():
    a = tensor((2, 2)).ones() * 2

    b = -a
    assert (b.isEqual(a * -1))


def test_add():
    a = tensor((2, 2)).ones() * 2

    a += a

    assert (a.isEqual(tensor((2, 2)).ones() * 4))


def test_add_tensor():
    a = tensor((2, 2)).ones() * 2
    b = tensor((2, 2)).ones() * 2
    c = a + b

    assert (c.isEqual(tensor((2, 2)).ones() * 4))
    assert not (c.isEqual(tensor((2, 2)).ones() * 2))


def test_mul_tensor():
    a = tensor((2, 2)).ones() * 2
    b = tensor((2, 2)).ones() * 2
    c = a * b

    assert (c.isEqual(tensor((2, 2)).ones() * 4))
    assert not (c.isEqual(tensor((2, 2)).ones() * 2))


def test_subtract_tensor():
    a = tensor((2, 2)).ones() * 2
    b = tensor((2, 2)).ones() * 2
    c = a - b

    assert (c.isEqual(tensor((2, 2)).zeros()))
    assert not (c.isEqual(tensor((2, 2)).ones() * 2))


def skip_test_divide_tensor():
    a = tensor((2, 2)).ones() * 2
    b = tensor((2, 2)).ones() * 2
    c = a / b

    assert (c.isEqual(tensor((2, 2)).ones()))

    c = a / (b * 0.5)

    assert (c.isEqual(tensor((2, 2)).ones() * 2))


def test_matmul():
    X = pare_array([
        [-2, 1],
        [0, 4]
    ])
    y = pare_array([
        [6, 5],
        [-7, 1]
    ])
    z = pare_array([
        [-19, -9],
        [-28, 4]
    ])
    assert z.isEqual(X.matmul(y))
    assert z.isEqual(X.matmul(y))
    assert z.isEqual(X.matmul(y))
    assert (z * 4).isEqual(X.matmul(y) * 4)

def test_transpose():
    X = pare_array([
        [-2, 1],
    ])
    z = pare_array([
        [-2],
        [1]
    ])
    assert z.isEqual(X.T())

    X = pare_array([
        [-2, 1],
        [2, -1],
    ])
    z = pare_array([
        [-2, 2],
        [1, -1]
    ])
    assert z.isEqual(X.T())


def test_tranpsposematmul():
    w = tensor((2, 3)).ones()
    v = tensor((3, 2)).ones()
    z = pare_array([
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
    ])
    z_p = v.matmul(w)
    assert z.isEqual(z_p)

    v_T = v.T()
    w_T = w.T()

    z = pare_array([
        [3, 3],
        [3, 3],
    ])
    z_p = v_T.matmul(w_T)
    z_p.print()
    assert z.isEqual(z_p)

