from cudaplayground import tensor,pare_array

def test_subtract():
    a = tensor((2, 2)).ones().cuda()

    b = a - 2

    b = b.host()
    a = a.host()

    assert (b.isEqual(tensor((2, 2)).ones() * (-1)))


def test_addition():
    a = tensor((2, 2)).zeros().cuda()
    b = a + 2

    b = b.host()
    a = a.host()

    assert (b.isEqual(tensor((2, 2)).ones() * (2)))


def test_mul():
    a = tensor((2, 2)).ones().cuda()
    b = -1 * (a * -1)

    b = b.host()
    a = a.host()

    assert (b.isEqual(tensor((2, 2)).ones()))

def test_divide():
    a = (tensor((2, 2)).ones() * 2).cuda()
    b = (4 / a).cuda()

    a = a.host()
    b = b.host()

    assert (b.isEqual(tensor((2, 2)).ones() * 2))

def test_transpose():
    X = pare_array([
        [-2, 1],
    ]).cuda()
    z = pare_array([
        [-2],
        [1]
    ])
    z_dot = X.T().host()

    z_dot.print()

    assert z.isEqual(z_dot)

    """
    X = pare_array([
        [-2, 1],
        [2, -1],
    ]).cuda()
    z = pare_array([
        [-2, 2],
        [1, -1]
    ])
    assert z.isEqual(X.T().host())
    """
