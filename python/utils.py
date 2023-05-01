import random
import numpy as np
import torch.cuda
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'



def apply_reward_to_go(raw_reward):
    # Reverse the list of rewards
    raw_reward = raw_reward[::-1]

    # Calculate the running sum of rewards
    running_sum = 0
    rtg_rewards = []
    for r in raw_reward:
        running_sum = r + running_sum
        rtg_rewards.append(running_sum)

    # Reverse the rtg_rewards list again to match the order of the original list
    rtg_rewards = rtg_rewards[::-1]

    # Normalize the rewards
    rtg_rewards = np.array(rtg_rewards)
    rtg_rewards = (rtg_rewards - np.mean(rtg_rewards)) / (np.std(rtg_rewards) + np.finfo(np.float32).eps)

    # Convert the rtg_rewards to a PyTorch tensor and return
    return torch.tensor(rtg_rewards, dtype=torch.float32, device=get_device())




def apply_discount(raw_reward, gamma=0.99):
    # Reverse the list of rewards
    raw_reward = raw_reward[::-1]

    # Calculate the discounted reward
    running_sum = 0
    discounted_rtg_reward = []
    for r in raw_reward:
        running_sum = r + gamma * running_sum
        discounted_rtg_reward.append(running_sum)

    # Reverse the discounted_rtg_reward list again to match the order of the original list
    discounted_rtg_reward = discounted_rtg_reward[::-1]

    # Normalize the rewards
    discounted_rtg_reward = np.array(discounted_rtg_reward)
    discounted_rtg_reward = (discounted_rtg_reward - np.mean(discounted_rtg_reward)) / (np.std(discounted_rtg_reward) + np.finfo(np.float32).eps)

    # Convert the discounted_rtg_reward to a PyTorch tensor and return
    return torch.tensor(discounted_rtg_reward, dtype=torch.float32, device=get_device())




# Util function to apply reward-return (cumulative reward) on a list of instant-reward (from eq 6)
def apply_return(raw_reward):
    # Compute r_reward (as a list) from raw_reward
    r_reward = [np.sum(raw_reward) for _ in raw_reward]
    return torch.tensor(r_reward, dtype=torch.float32, device=get_device())