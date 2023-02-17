
def get_x_y():
    sentene_length = 25
    content = None
    with open("phrack.txt", "r") as file:
        content = file.read().replace("\n", " ").strip()

    X = []
    y = []
    for i in range(0, len(content)):
        x = content[i:i+sentene_length].lower()
        if len(x) < 5:
            continue
        X.append(x)
        y.append("".join(list(reversed(list(x)))))
    
    return X, y
