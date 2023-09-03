from train_dreamer_v1 import get_training_data

sum_reward = 0
for reward  in get_training_data().tensors[-1]:
    sum_reward += reward
    print(1)
print(sum_reward.item())

