"""
Bad implementation, don't use in production. Just a poc.
"""

from collections import defaultdict

class ByteStream:
    def __init__(self, chars) -> None:
        self.chars = self._split(chars)
    
    def merge(self, pair):
        new_chars = []
        i = 0
        while i < (len(self.chars) - 1):
            if (self.chars[i], self.chars[i + 1]) == pair:
                new_chars.append(self.chars[i] + self.chars[i + 1])
                i += 1
            else:
                new_chars.append(self.chars[i])
            i += 1
        self.chars = new_chars
    
    def _split(self, chars):
        return sum([
            list(i) + ["</w>", " "] for i in chars
        ], [])

    def __str__(self):
        return str(self.chars)

    def __repr__(self) -> str:
        return self.__str__()

class Example_BPE:
    def __init__(self) -> None:
        pass

    def encode(self, sentences, iterations=5):
        encoding = []
        for i in sentences:
            encoding.append(ByteStream(i.split(" ")))
        
        for i in range(iterations):
            frequency = defaultdict(int)
            for i in encoding:
                print(i.chars)
                for index in range(len(i.chars) - 1):
                    pair = tuple((i.chars[index], i.chars[index + 1]) )
                    if "</w>" in pair:
                        continue
                    frequency[pair] += 1
            print("")
            most_used = max(frequency.items(), key=lambda x: x[1])[0]
            if "</w>" in most_used:
                return
        
            for i in encoding:
                i.merge(most_used)

        return encoding

sentences = [
    "test av festen",
    "festen er ikke bra",
    "testen gikk veldig bra"
]

encoding = Example_BPE().encode(sentences)
print(encoding[0])
"""
Then you just make a table that maps the tokens as normal. Win win.

By doing this you should be able to compress down the vocab size.
"""
