from .simpel_model_interface import SimpleModelInterface
from .train import train_model

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
    
    train_model(
        model,
        model.get_training(
            [
                "Hello sir, I hope you are having a good day",
                "Good night, I hope we meet again tomorrow!"
            ]
        )
    )
    output = model.forward(
        "Hello sir"
    )
    print(output)
    output = model.forward(
        "I hope"
    )
    print(output)
    