import torchvision.transforms as T

def deletion(f, image, s, N):
    n = [0]
    image = image.clone()
    h_n = [f(image)]

    coordinates_value = get_sorted_importance(s)
    while len(coordinates_value):
        for _ in range(min(N, len(coordinates_value))):
            x, y, _ = coordinates_value.pop()
            image[x][y] = 0

        n.append(n[-1] + 1)
        h_n.append(f(image).item())

    return n, h_n

def insertion(f, image, s, N):
    blur = T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))
    I_dot = blur(image.reshape((1, ) + image.shape))[0]
    n = [0]
    h_n = [f(I_dot)]

    coordinates_value = get_sorted_importance(s)
    while len(coordinates_value):
        for _ in range(min(N, len(coordinates_value))):
            x, y, _ = coordinates_value.pop()
            I_dot[x][y] = image[x][y]

        n.append(n[-1] + 1)
        h_n.append(f(I_dot).item())

    return n, h_n

def get_sorted_importance(s):
    coordinates_value = []
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            coordinates_value.append((i, j, s[i][j].item()))

    coordinates_value = sorted(coordinates_value, key=lambda x: x[-1], reverse=True)
    return coordinates_value
