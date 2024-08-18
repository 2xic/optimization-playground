"""
https://en.wikipedia.org/wiki/Xorshift
https://stackoverflow.com/a/71523041

Found this in https://github.com/EurekaLabsAI/ngram/blob/master/ngram.py
"""

class XorShift:
    def __init__(self) -> None:
        self.state = 1
        self.max_digit = 2048

    def xorshift_star(self):
        # https://en.wikipedia.org/wiki/Xorshift#xorshift*
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        return self.state * 0x2545F4914F6CDD1D

if __name__ == "__main__":
    a = XorShift()
    for _ in range(100):
        print(a.xorshift_star())

