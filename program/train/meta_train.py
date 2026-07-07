import os
import sys
import argparse
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from gymnasium import spaces

from world import World
from env import TSCEnv
from agent.meta_dynamics_agent import MetaDynamicsAgent
from utils.computer_monitor import computer_monitor


parser = argparse.ArgumentParser()
parser.add_argument('--config1', type=str, required=True)
parser.add_argument('--config2', type=str, required=True)
parser.add_argument('--thread_num', type=int, default=20, help='the number of thread')
parser.add_argument('--max_episode', type=int, default=200, help='maximum episode')
parser.add_argument('--max_step', type=int, default=180)
parser.add_argument('--stride', type=int, default=20)
parser.add_argument('--save_model_freq', type=int, default=20)
parser.add_argument('--save_model_dir', type=str, default='outputs/models/meta_dynamics/4x4_and_28x7/')
parser.add_argument('--load_model_dir', type=str, default='')

args = parser.parse_args()

# build world
world_1 = World(config_path=args.config1, thread_num=args.thread_num, max_time=args.max_step*args.stride)
world_2 = World(config_path=args.config2, thread_num=args.thread_num, max_time=args.max_step*args.stride)

agents_1 = []
for idx, inter_id in enumerate(world_1.inter_list):
    action_space = spaces.Discrete(8)
    agent = MetaDynamicsAgent(world_1, idx, 'observation', action_space)
    agents_1.append(agent)
    
agents_2 = []
for idx, inter_id in enumerate(world_2.inter_list):
    action_space = spaces.Discrete(8)
    agent = MetaDynamicsAgent(world_2, idx, 'observation', action_space)
    agents_2.append(agent)
agent_shared = MetaDynamicsAgent(world_1, -1, 'decision', spaces.Discrete(8))

env_1 = TSCEnv(world_1, agents_1)
env_2 = TSCEnv(world_2, agents_2)

# train
def train(args, env_1, env_2):
    exp_num = 0
    total_mean_episode_rewards_1 = []
    total_mean_episode_rewards_2 = []
    for episode in range(args.max_episode):
        computer_monitor()
        last_obs_1, last_states_1 = env_1.reset()
        last_obs_2, last_states_2 = env_2.reset()
        episode_rewards_1 = [0 for _ in range(len(agents_1))]
        episode_rewards_2 = [0 for _ in range(len(agents_2))]
        cur_step = 0
        while cur_step < args.max_step:
            actions_1 = []
            for idx in range(len(agents_1)):
                if exp_num >= agent_shared.learning_start:
                    actions_1.append(agent_shared.choose_action(last_states_1[idx], 'train'))
                else:
                    actions_1.append(agent_shared.sample())
            
            actions_2 = []
            for idx in range(len(agents_2)):
                if exp_num >= agent_shared.learning_start:
                    actions_2.append(agent_shared.choose_action(last_states_2[idx], 'train'))
                else:
                    actions_2.append(agent_shared.sample())
    
            rewards_list_1 = []
            rewards_list_2 = []
            obs_1, dones_1, states_1, infos_1 = None, None, None, None
            obs_2, dones_2, states_2, infos_2 = None, None, None, None
            for i in range(args.stride):
                if i == args.stride - 1:
                    env_1.stat_through_cnt()
                    env_2.stat_through_cnt()
                obs_1, rewards_1, dones_1, infos_1 = env_1.step(actions_1)
                obs_2, rewards_2, dones_2, infos_2 = env_2.step(actions_2)
                states_1 = infos_1['states']
                states_2 = infos_2['states']
                rewards_list_1.append(rewards_1)
                rewards_list_2.append(rewards_2)
            cur_step += 1
            mean_rewards_1 = np.mean(rewards_list_1, axis=0)
            mean_rewards_2 = np.mean(rewards_list_2, axis=0)

            for idx in range(len(agents_1)):
                agent_shared.remember(None, last_states_1[idx], actions_1[idx], mean_rewards_1[idx], states_1[idx])
                episode_rewards_1[idx] += mean_rewards_1[idx]
                exp_num += 1

            for idx in range(len(agents_2)):
                agent_shared.remember(None, last_states_2[idx], actions_2[idx], mean_rewards_2[idx], states_2[idx])
                episode_rewards_2[idx] += mean_rewards_2[idx]
                exp_num += 1

            last_obs_1 = obs_1
            last_states_1 = states_1

            last_obs_2 = obs_2
            last_states_2 = states_2

            if exp_num >= agent_shared.learning_start and cur_step % agent_shared.update_model_freq == 0:
                agent_shared.meta_replay()
                
            if all(dones_1) or all(dones_2):
                break
        
        if episode % args.save_model_freq == args.save_model_freq - 1:
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir, exist_ok=True)
            agent_shared.save_model(args.save_model_dir)
        
        total_mean_episode_reward_1 = np.mean([(episode_reward_1 / args.max_step) for episode_reward_1 in episode_rewards_1])
        total_mean_episode_rewards_1.append(total_mean_episode_reward_1)
        print(f'episode {episode} rewards: {total_mean_episode_reward_1}')

        total_mean_episode_reward_2 = np.mean([(episode_reward_2 / args.max_step) for episode_reward_2 in episode_rewards_2])
        total_mean_episode_rewards_2.append(total_mean_episode_reward_2)
        print(f'episode {episode} rewards: {total_mean_episode_reward_2}')
        
def load_model(model_dir):
    dynamics_model = os.path.join(model_dir, 'dynamics_model.pth')
    agent_shared.load_model(dynamics_model)


if __name__ == '__main__':
    if args.load_model_dir:
        load_model(args.load_model_dir)
    train(args, env_1, env_2)
