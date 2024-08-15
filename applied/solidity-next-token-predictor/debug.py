from torch_gpt_like_model_bigger import get_cache_file, get_cached_model, SEQUENCE_LENGTH
import torch

if __name__ == "__main__":
    # todo: vocab needs to be pre-generated on the dataloader side.
    vocab = get_cache_file()
    assert vocab is not None
    print("Loaded vocab")
    model = get_cached_model(vocab).eval()

    # Predict the next tokens for contract keyword
    text = vocab.get_tensor(
        "contract ",
        SEQUENCE_LENGTH
    )
    output = (model.rollout(text[0], 10, torch.device("cpu")))
    print(vocab.decode(output))
