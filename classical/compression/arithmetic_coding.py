"""
From paper 
https://www.cs.cmu.edu/~aarti/Class/10704/Intro_Arith_coding.pdf
"""

DEBUG = False

class ByteCodes:
    def __init__(self, value) -> None:
        value = float(value)
        self.raw = value
        self.value = f"{value:.8f}"[2:]
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

    def sub(self, b):
        a = max(len(b.value), len(self.value))
       # print((b.value, self.value))
        c = ""
        for i in range(a - 1, -1, -1):
            a_i = self.value[i] if i < len(self.value) else 0
            b_i = b.value[i] if i < len(b.value) else 0
            #c_i = counter - int(a_i) - int(b_i) 
            #counter = int(c_i > 1)
            c_i = None
            if a_i == '0' and b_i == '0':
                c_i = 0
            elif a_i == '1' and b_i == '0':
                c_i = 1
            elif a_i == '0' and b_i == '1':
                # need to borrow
                for ii in range(i, -1, -1):
                    #print(self.value[i])
                    if (ii < len(self.value) and str(self.value[ii]) == '1'):
                        c_i = 1
                        self.value = list(self.value)
                        self.value[ii] = '0'
                        print(self.value[ii])
                        break
                #if c_i is None:
                c_i = 1
            elif a_i == '1' and b_i == '1':
                c_i = 0
            if c_i is None:
                print((i, a_i, b_i, c_i))
                raise Exception("Error")
            c = str(c_i) + str(c)
           # print(c)
        #print("")
        return ByteCodes("0." + c)

    def __str__(self):
        return f"{self.value} ({self.raw})"

    def __repr__(self):
        return self.__str__()

if DEBUG:
    a = ByteCodes("0.011")
    b = ByteCodes("0.0011")
    assert a.add(b).value == "10010000", a.add(b).value

    a = ByteCodes("0.1010011")
    b = ByteCodes("0.0110000")
    assert a.sub(b).value == "01000110", a.add(b).value

    a = ByteCodes("0.011011")
    b = ByteCodes("0.0110")
    assert a.sub(b).value == "001011", a.add(b).value

    a = ByteCodes("0.100011")
    b = ByteCodes("0.0110000")
    assert a.sub(b).value == "01101100", a.add(b).value

#exit(0)
#exit(0)
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
# encoding the string
def encode(string):
    c = ByteCodes(0)
    a = 1

    for i in string:
#        print(c.raw, a, symbols_cdf[i].raw, a * symbols_cdf[i].raw)
        print(f"c = {c.raw}, a = {a}")
        c = c.add(ByteCodes(a * symbols_cdf[i].raw))
        print(f"symbol_cdf={symbols_cdf[i].raw}")
#        print(f"current code worth * a = {ByteCodes(a * symbols_cdf[i].raw).raw}")
        a = round(a * symbols_prob[i], 8)
        print(f"after c = {c.raw}, a = {a}")
        print("")
    return c, a

"""
0.011       # a cdf
   011      # a cdf
    001     # b cdf
      111   # c cdf
0.1010011
"""
c,a = encode("aabc")
# decode from the interval
def decode(c):
    """
    Examine the code string and determine the interval in whichit lies. Decode the symbol
    corresponding to that interval.
    
    Since the first subinterval is [0.011, 0.111] we know we got an A
    [cdf, cdf + p] 
    """ 
    output = ""
    for _ in range(3):
        symbols = sorted(symbols_cdf.keys(), key=lambda x: symbols_cdf[x].raw)
        for index, i in enumerate(symbols):
          #  print(i)
            from_interval = symbols_cdf[i].raw
            to_interval = symbols_cdf[i].raw + symbols_prob[i]
            if from_interval <= c.raw and c.raw <= to_interval:
                output += i
                # we know what is added, and we can therefore subtract
                c = c.sub(ByteCodes(from_interval))
                c = c.add(c)
                # this process is repeated for all symbols
                break
    print(output)

decode(c)


