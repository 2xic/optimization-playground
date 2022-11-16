import torch

def deletion(f, image, s, N):
    n = [0]
    h_n = [f(image)]

    coordinates_value = []
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            #if s[i][j] > 0:
            coordinates_value.append((i, j, s[i][j].item()))
    coordinates_value = sorted(coordinates_value, key=lambda x: x[-1], reverse=True)
    while len(coordinates_value):
        for _ in range(min(N, len(coordinates_value))):
            x, y, c = coordinates_value.pop()
            image[x][y] = 0

        n.append(n[-1] + 1)
        h_n.append(f(image).item())

    return n, h_n
