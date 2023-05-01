import pickle
import gymnasium as gym
import torch
from gymnasium.utils.save_video import save_video
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
from utils import *


class PGTrainer:
    def __init__(self, params):
        self.params = params
        self.env = gym.make(self.params['env_name'])
        self.agent = Agent(env=self.env, params=self.params)
        self.actor_policy = PGPolicy(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.optimizer = Adam(params=self.actor_policy.parameters(), lr=self.params['lr'])

    def run_training_loop(self):
        list_ro_reward = list()

        for ro_idx in range(self.params['n_rollout']):
            trajectory = self.agent.collect_trajectory(policy=self.actor_policy)
            loss = self.estimate_loss_function(trajectory)
            self.update_policy(loss)
            # Calculate the average reward for each trajectory
      
            total_ro_reward = 0
            ntr = self.params['n_trajectory_per_rollout']
            for tr_idx in range(ntr):
                start_idx = tr_idx * self.params['n_rollout']
                end_idx = start_idx + self.params['n_rollout']
                total_ro_reward += sum([sum(r) for r in trajectory['reward'][start_idx:end_idx]])

            avg_ro_reward = total_ro_reward / ntr
            print(f'End of rollout {ro_idx}: Average trajectory reward is {avg_ro_reward: 0.2f}')
            # Append average rollout reward into a list
            list_ro_reward.append(avg_ro_reward)
        # Save avg-rewards as pickle files
        pkl_file_name = self.params['exp_name'] + '.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(list_ro_reward, f)
        # Save a video of the trained agent playing
        self.generate_video()
        # Close environment
        self.env.close()

    
    def estimate_loss_function(self, trajectory):
        loss = []
        for traj_idx in range(self.params['n_trajectory_per_rollout']):
        # Get the rewards for the current trajectory
            reward = torch.tensor(trajectory['reward'][traj_idx], dtype=torch.float32)
        
        # Get the log-probs for the current trajectory
            log_prob = trajectory['log_prob'][traj_idx]
        
        # Compute the loss based on the flags
            if self.params['reward_to_go'] and self.params['reward_discount']:
                rtg_reward = apply_reward_to_go(reward.tolist)
                discounted_reward = apply_discount(reward.tolist())
                rtg_discounted_reward = apply_discount(rtg_reward)
                loss.append(-1 * (rtg_discounted_reward * log_prob).mean())
            elif self.params['reward_to_go']:
                rtg_reward = apply_reward_to_go(reward.tolist())
                loss.append(-1 * (rtg_reward * log_prob).mean())
            elif self.params['reward_discount']:
                discounted_reward = apply_discount(reward.tolist())
                loss.append(-1 * (discounted_reward * log_prob).mean())
            else:
                r_reward = apply_return(reward.tolist())
                loss.append(-1 * (r_reward * log_prob).mean())
        loss = torch.stack(loss).mean()
        return loss




    def update_policy(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def generate_video(self, max_frame=1000):
        env_name = self.params['env_name']
        exp_name = self.params['exp_name']
        self.env = gym.make(env_name, render_mode='rgb_array_list')
        obs, _ = self.env.reset()
        for _ in range(max_frame):
            action_idx, log_prob = self.actor_policy(torch.tensor(obs, dtype=torch.float32, device=get_device()))
            obs, reward, terminated, truncated, info = self.env.step(self.agent.action_space[action_idx.item()])
            if terminated or truncated:
                break
        video_name = f"{exp_name}_{env_name[:-3]}"
        save_video(frames=self.env.render(), video_folder=video_name, fps=self.env.metadata['render_fps'], step_starting_index=0, episode_index=0)
        self.env.close()

class PGPolicy(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(PGPolicy, self).__init__()
        # Define the policy net
        self.policy_net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        # Forward pass of policy net
        policy_dist = Categorical(self.policy_net(obs))
        action_index = policy_dist.sample()
        log_prob = policy_dist.log_prob(action_index)
        return action_index, log_prob


class Agent:
    def __init__(self, env, params=None):
        self.env = env
        self.params = params
        self.action_space = [action for action in range(self.env.action_space.n)]

    def collect_trajectory(self, policy):
        obs, _ = self.env.reset(seed=self.params['rng_seed'])
        rollout_buffer = list()
        for _ in range(self.params['n_trajectory_per_rollout']):
            trajectory_buffer = {'log_prob': list(), 'reward': list()}
            while True:
                # Get action from the policy (forward pass of policy net)
                action_idx, log_prob = policy.forward(torch.from_numpy(obs).float())
                obs, reward, terminated, truncated, info = self.env.step(self.action_space[action_idx.item()])
                # Save log-prob and reward into the buffer
                trajectory_buffer['log_prob'].append(log_prob)
                trajectory_buffer['reward'].append(reward)
                # Check for termination criteria
                if terminated or truncated:
                    obs, _ = self.env.reset(seed=self.params['rng_seed'])
                    rollout_buffer.append(trajectory_buffer)
                    break
        rollout_buffer = self.serialize_trajectory(rollout_buffer)
        return rollout_buffer

    # Converts a list-of-dictionary into dictionary-of-list
    @staticmethod
    def serialize_trajectory(rollout_buffer):
        serialized_buffer = {'log_prob': list(), 'reward': list()}
        for trajectory_buffer in rollout_buffer:
            serialized_buffer['log_prob'].append(torch.stack(trajectory_buffer['log_prob']))
            serialized_buffer['reward'].append(trajectory_buffer['reward'])
        return serialized_buffer