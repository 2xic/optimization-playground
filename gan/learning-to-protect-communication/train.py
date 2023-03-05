from model import Eve, Alice, Bob
import torch
from utils import loss_eve, create_n_bits_entry, loss_alice_bob, test_encrypt
import torch.optim as optim

BATCH_SIZE = 256
N = 8

eve = Eve(
    plaintext=N,
)
alice = Alice(
    plaintext=N,
    sharedkey=N
)
bob = Bob(
    plaintext=N,
    sharedkey=N
)

eve_optimizer = optim.Adam(
    eve.parameters(),
    lr=0.0008
)
alice_bob = optim.Adam(
    list(alice.parameters()) + list(bob.parameters()),
    lr=0.0008
)

for i in range(10_000):
    plain = create_n_bits_entry(BATCH_SIZE, N)
    shared = create_n_bits_entry(BATCH_SIZE, N)

   # print(plain)

    eve_optimizer.zero_grad()
    eve_error = loss_eve(
        plain,
        shared,
        alice=alice,
        eve=eve
    )
    eve_error.backward()
    eve_optimizer.step()

    alice_bob.zero_grad()
    alice_bob_error = loss_alice_bob(
        plain,
        shared,
        alice=alice,
        bob=bob,
        eve=eve
    )
    alice_bob_error.backward()
    alice_bob.step()

    if i % 1_000 == 0:
        print(f"Error for eve {eve_error.item()}")
        print(f"Error for alice, bob {alice_bob_error.item()}")

        with torch.no_grad():
            test_encrypt(
                torch.tensor([-1, 1, ] * (N // 2)).reshape((1, -1)),
                create_n_bits_entry(1, N),
                alice=alice,
                bob=bob,
                eve=eve
            )
        
        print("")
    