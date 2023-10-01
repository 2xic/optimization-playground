"""
From paper 
https://www.cs.cmu.edu/~aarti/Class/10704/Intro_Arith_coding.pdf
"""

class ByteCodes:
    def __init__(self, value) -> None:
        value = float(value)
        self.raw = value
        self.value = f"{value:.12f}"[2:]
    #    print(self.value)

    def add(self, b):
        a = max(len(b.value), len(self.value))
       # print((b.value, self.value))
        c = ""
        counter = 0
        for i in range(a - 1, -1, -1):
            a_i = self.value[i] if i < len(self.value) else 0
            b_i = b.value[i] if i < len(b.value) else 0
            c_i = counter + int(a_i) + int(b_i) 
            counter = int(c_i > 1)
            c = str(c_i % 2) + str(c)
        return ByteCodes("0." + c)
a = ByteCodes("0.011")
b = ByteCodes("0.0011")
assert a.add(b).value == "100100000000", a.add(b).value

#exit(0)

symbols_cdf = {
    "a": ByteCodes(0.011),
    "b": ByteCodes(0.001),
    "c": ByteCodes(0.111),
    "d": ByteCodes(0.000),
}
symbols_prob = {
    "a": 0.100,
    "b": 0.010,
    "c": 0.001,
    "d": 0.001,
}
def encode(string):
    c = ByteCodes(0)
    a = 1

    for i in string:
        print(c.raw, a, symbols_cdf[i].raw, a * symbols_cdf[i].raw)
        c = c.add(ByteCodes(a * symbols_cdf[i].raw))
        a = a * symbols_prob[i]
        print(c.raw, f"{a:.8f}")
        print("")

encode("aabd")
