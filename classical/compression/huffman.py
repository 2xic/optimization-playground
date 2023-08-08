import heapq

class Symbol:
    def __init__(self, char) -> None:
        self.char = char
        self.frequency = 1

    def inc(self):
        self.frequency += 1

    def __str__(self) -> str:
        return f"{self.char} (count: {self.frequency})"

    def __repr__(self) -> str:
        return self.__str__()
    
class Node:
    def __init__(self, symbol) -> None:
        self.symbol = symbol
        self.left = None
        self.right = None
        self.frequency = symbol.frequency if symbol is not None else 0
        self.code = ''

    def __lt__(self, other):
        return self.frequency < other.frequency

frequency = {}
text = "hello sir, this is just a test"

for i in text:
    if i in frequency:
        frequency[i].inc()
    else:
        frequency[i] = Symbol(i)

nodes = []
for i in frequency:
    heapq.heappush(nodes, Node(frequency[i]))

"""
Construct the tree
"""
while 1 < len(nodes):
    left = heapq.heappop(nodes)
    right = heapq.heappop(nodes)

    left.code = '0'
    right.code = '1'

    merged = Node(None)
    merged.frequency = left.frequency + right.frequency
    merged.left = left
    merged.right = right

    heapq.heappush(nodes, merged)

stack = [(nodes[0], '', nodes[0].symbol)]
while len(stack):
    (node, code, symbol) = stack.pop(0)
    if node.left is not None:
        stack.append((node.left, code + node.left.code, node.left.symbol))
    if node.right is not None:
        stack.append((node.right, code + node.right.code, node.right.symbol))
    
    if node.left is None and node.right is None:
        print(f"{symbol}: {code}")
