import torch
import torch.nn as nn

def distance(plain: torch.Tensor, decrypted: torch.Tensor):
    # l1 distance is used in the paper
    loss = torch.nn.L1Loss()
    return loss(
        decrypted, plain
    )

def adjusted_eve_loss(N: int, eve: torch.Tensor):
    return (
        N / 2 - eve
    ) ** 2 / (
        N / 2 
    ) ** 2

def loss_eve(plain, shared, alice, eve):
    encrypted = alice(plain, shared)#.detach()
    return distance(
        plain, eve(encrypted)
    )

def loss_bob(plain, shared, alice, bob):
    encrypted = alice(plain, shared)
    return distance(
        plain , bob(encrypted, shared)
    )

def loss_alice_bob(plain, shared, alice, bob, eve):
#    return eve(alice(plain, shared).detach())
    loss_decoding = loss_bob(
        plain, shared, alice, bob
    )
    loss_eve_raw = loss_eve(
        plain, shared, alice, eve
    )#.detach()

    loss_eve_scaled = adjusted_eve_loss(
        plain.shape[-1],
        loss_eve_raw
    )
    # since we are scaling up the loss
    # the loss of eve > our output
    # I think this sends the wrong message to the model?
    # or maybe not, maybe we just need the model to fool eve more.
    return (loss_decoding  + (1 - loss_eve_raw ** 2)), loss_decoding
#    return (loss_decoding  - loss_eve_scaled), loss_decoding

def create_n_bits_entry(batch, N):
    plain = torch.randint(low=-1, high=1, size=(batch, N))
    plain[plain < 0] = -1
    plain[plain >= 0] = 1
    return plain.float()


def test_encrypt(plain_text, shared_key, alice, bob, eve):
    encrypted = alice(plain_text, shared_key)
    decrypted = bob(encrypted, shared_key)

    # print(decrypted)
    # print(_simple_tensor(decrypted))
    
    eve_decrypted = eve(encrypted)

    bits_wrong = (
        calculate_bits_wrong(plain_text[0], eve_decrypted[0].clone()),
        calculate_bits_wrong(plain_text[0], decrypted[0].clone())
    )

    print(f"Input {_simple_tensor(plain_text)}")
    print(f"Encrypted {_simple_tensor(encrypted)}")
    print(f"Decrypted {_simple_tensor(decrypted)}")
    print(f"Reconstructed {_simple_tensor(eve_decrypted)}")
    print(f"Eve bits wrong {bits_wrong[0]}")
    print(f"Bob bits wrong {bits_wrong[1]}")
    print("")
    return bits_wrong

def _simple_tensor(tensor: torch.Tensor):
    return [
        round(i, 2) for i in tensor.tolist()[0]
    ][:10]

def calculate_bits_wrong(expected: torch.Tensor, predicted: torch.Tensor):
    predicted = round_to_bits(predicted)

    assert len(predicted.shape) == 1, "Expected single dimension"
    correct_bits = ((predicted == expected).sum())
    return expected.shape[-1] - correct_bits

def round_to_bits(raw_tensor):
    raw_tensor[raw_tensor < 0] = -1
    raw_tensor[raw_tensor >= 0] = 1
    return raw_tensor

def clip_gradients(model, clip):
    nn.utils.clip_grad_value_(model.parameters(), clip)
