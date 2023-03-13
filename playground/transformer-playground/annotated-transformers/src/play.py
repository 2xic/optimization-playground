from .simpel_model_interface import SimpleModelInterface


if __name__ == "__main__":
    model = SimpleModelInterface()
    model.build_model([
        "Hello sir, I hope you are having a good day",
        "Good night, I hope we meet again tomorrow!"
    ])
    output = model.forward(
        "Hello sir"
    )
    print(output)
    