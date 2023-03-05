from model import Eve, Alice, Bob
import torch
from utils import loss_eve, create_n_bits_entry, loss_alice_bob, test_encrypt, clip_gradients
import torch.optim as optim
import matplotlib.pyplot as plt
import os

for runs in range(20): 
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # The authors said they used this batch size (if not otherwise stated)
    # They also mentioned they got more stable results with batch size of 512
    #BATCH_SIZE = 4096
    BATCH_SIZE = 512


    # N = 16 was used for the Figure 2. 
    N = 16
    # In the paper things only start to happen at around 15 000 steps
    STEPS = 25_000
    #STEPS = 100_000

    # Same as in the paper, and it's not reduced over time
    LR = 0.0008
    # Batch loops
    ALICE_BOB_MINI_BATCHES_PER_STEP = 1
    # Paper has EVE_MINI_BATCHES_PER_STEP = 2 to give Eve a small edge
    EVE_MINI_BATCHES_PER_STEP = 2

    CLIP_VALUE = 1

    """
    The paper also notes that training this model is a bit challenging 6/20 runs failed (
        failed meaning either bob not being able to decode message, or eve being able to decode message
    ).
    """

    eve = Eve(
        plaintext=N,
    ).to(DEVICE)
    alice = Alice(
        plaintext=N,
        sharedkey=N
    ).to(DEVICE)
    bob = Bob(
        plaintext=N,
        sharedkey=N
    ).to(DEVICE)

    eve_optimizer = optim.Adam(
        eve.parameters(),
        lr=LR
    )
  #  alice_bob = optim.Adam(
  #      list(alice.parameters()) + list(bob.parameters()),
  #      lr=LR
  #  )
    alice_optimizer = optim.Adam(
        list(alice.parameters()),
        lr=LR
    )
    bob_optimizer = optim.Adam(
        list(bob.parameters()),
        lr=LR
    )

    x = []
    eve_bits_wrong_y = []
    bob_bits_wrong_y = []

    alice_bob_loss_y = []
    bob_loss_y = []
    eve_loss_y = []

    mean_value_bob = []
    mean_value_alice = []
    mean_value_eve = []

    for i in range(STEPS):
        eve_error = None  
        for _ in range(EVE_MINI_BATCHES_PER_STEP):
            plain = create_n_bits_entry(BATCH_SIZE, N).to(DEVICE)
            shared = create_n_bits_entry(BATCH_SIZE, N).to(DEVICE)
            eve_optimizer.zero_grad()
            eve_error = loss_eve(
                plain,
                shared,
                alice=alice,
                eve=eve
            )
            eve_error.backward()
            clip_gradients(eve, CLIP_VALUE)
            eve_optimizer.step()

        for step_index in range(ALICE_BOB_MINI_BATCHES_PER_STEP):
            plain = create_n_bits_entry(BATCH_SIZE, N).to(DEVICE)
            shared = create_n_bits_entry(BATCH_SIZE, N).to(DEVICE)

            alice_optimizer.zero_grad()
            bob_optimizer.zero_grad()
            (alice_bob_error, bob_error) = loss_alice_bob(
                plain,
                shared,
                alice=alice,
                bob=bob,
                eve=eve
            )
            alice_bob_error.backward()
            alice_optimizer.step()
            clip_gradients(alice, CLIP_VALUE)
            clip_gradients(bob, CLIP_VALUE)
            bob_optimizer.step()

            if i % 1_000 == 0 and step_index == 0:
                print(f"Run {runs}: {i} / {STEPS}")
                print(f"Error for eve {eve_error.item()}")
                print(f"Error for bob {bob_error.item()}")
                print(f"Error for alice, bob {alice_bob_error.item()}")

                eve_loss_y.append(eve_error.item())
                bob_loss_y.append(bob_error.item())
                alice_bob_loss_y.append(alice_bob_error.item())

                alice_output = alice(plain, shared)
                mean_value_bob.append((bob(alice_output, shared)).mean().item())
                mean_value_alice.append(alice_output.mean().item())
                mean_value_eve.append(eve(alice_output).mean().item())

                with torch.no_grad():
                    #test_tensor = torch.tensor([-1, 1, ] * (N // 2)).reshape((1, -1)).to(DEVICE)
                    #test_shared_key = create_n_bits_entry(1, N).to(DEVICE)
                    test_tensor = plain[0].reshape((1, -1))
                    test_shared_key = shared[0].reshape((1, -1))
                    (eve_bits_wrong, bob_bits_wrong) = test_encrypt(
                        test_tensor,
                        test_shared_key,
                        alice=alice,
                        bob=bob,
                        eve=eve
                    )
                    x.append(
                        i
                    )
                    eve_bits_wrong_y.append(
                        eve_bits_wrong.item()
                    )
                    bob_bits_wrong_y.append(
                        bob_bits_wrong.item()
                    )
    output_path = lambda x: os.path.join(os.path.dirname(os.path.abspath(__file__)), f"runs/{runs}/", x)
    os.makedirs(
        os.path.dirname(output_path('reconstruction_error.png')),
        exist_ok=True
    )
    plt.plot(x, eve_bits_wrong_y, color="green", label="eve")
    plt.plot(x, bob_bits_wrong_y, color="red", label="bob")
    plt.legend(loc="upper left")
    plt.ylabel("Bits wrong")
    plt.savefig(output_path('reconstruction_error.png'))
    plt.clf()

    plt.plot(x, eve_loss_y, color="green", label="eve")
    plt.plot(x, bob_loss_y, color="red", label="bob")
    plt.plot(x, alice_bob_loss_y, color="yellow", label="bob - eve")
    plt.legend(loc="upper left")
    plt.ylabel("Loss")
    plt.savefig(output_path('loss.png'))
    plt.clf()

    plt.plot(x, mean_value_eve, color="green", label="eve")
    plt.plot(x, mean_value_bob, color="red", label="bob")
    plt.plot(x, mean_value_alice, color="yellow", label="alice")
    plt.legend(loc="upper left")
    plt.ylabel("Mean output value")
    plt.savefig(output_path('output.png'))
    plt.clf()
