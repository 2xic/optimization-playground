import torch

def loss_eve(plain, shared, alice, eve):
    return (
        plain - eve(alice(plain, shared).detach())
    ).mean()

def loss_alice_bob(plain, shared, alice, bob, eve):
#    return eve(alice(plain, shared).detach())
    loss_decoding = loss_bob(
        plain, shared, alice, bob
    )
    loss_secret = loss_eve(
        plain, shared, alice, eve
    ).detach()

    return loss_decoding - loss_secret

def loss_bob(plain, shared, alice, bob):
    return (
        plain - bob(alice(plain, shared), shared)
    ).mean()

def create_n_bits_entry(batch, N):
    plain = torch.randint(low=-1, high=1, size=(batch, N))
    plain[plain < 0] = -1
    plain[plain >= 0] = 1
    return plain.float()


def test_encrypt(plain_text, shared_key, alice, bob, eve):
    encrypted = alice(plain_text, shared_key)
    decrypted = bob(encrypted, shared_key)

    eve_decrypted = eve(encrypted)

    print(f"Input {_simple_tensor(plain_text)}")
    print(f"Encrypted {_simple_tensor(encrypted)}")
    print(f"Decrypted {_simple_tensor(decrypted)}")
    print(f"Reconstructed {_simple_tensor(eve_decrypted)}")

def _simple_tensor(tensor: torch.Tensor):
    return [
        round(i, 2) for i in tensor.tolist()[0]
    ][:10]