"""
Based on
- https://alemorales.info/post/automatic-differentiation-with-dual-numbers/
"""


class Value:
    def __init__(self, value, deriv) -> None:
        self.value = float(value)
        self.deriv = float(deriv)

    def __mul__(self, other):
       # print(
        #    (other.value, self.value),
         #   (other.deriv, self.deriv)
        #)
        return Value(
            self.value * other.value,
            other.value * self.deriv +
            other.deriv * self.value
        )

    def __add__(self, other):
        return Value(
            self.value + other.value,
            self.deriv + other.deriv
        )

    def __pow__(self, number):
        return Value(
            self.value ** number,
            number * self.value ** (number - 1) * self.deriv
        )


if __name__ == "__main__":
    x = Value(1.0, 1.0)
    y = Value(0.5, 0.0)
    z = Value(2.0, 0.0)

    q = z * (x + y) ** 2
    assert q.value == 4.5
    # &d/&x
    assert q.deriv == 6
