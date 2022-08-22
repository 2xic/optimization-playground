
from cfd import CFD


cfd = CFD()

# rock always win 
# https://www.youtube.com/watch?v=b0SoKWLkmLU
strategy = cfd.train([1, 0, 0])
print(strategy.round(3))
