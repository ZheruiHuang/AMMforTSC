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
from agent.non_modular_agent import NonModularAgent
from utils.computer_monitor import computer_monitor


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--mode', type=str, choices=('finetune', 'meta_train'), default='meta_train')
parser.add_argument('--model_dir', type=str, default='')
parser.add_argument('--isPart', action='store_true', default=False)
parser.add_argument('--episode', type=int, default=1)
parser.add_argument('--save_model_freq', type=int, default=1)
parser.add_argument('--save_model_dir', type=str, default='outputs/models/non_modular/')

parser.add_argument('--thread_num', type=int, default=20)
parser.add_argument('--max_step', type=int, default=180)
parser.add_argument('--stride', type=int, default=20)

args = parser.parse_args()

# build world
world = World(config_path=args.config, thread_num=args.thread_num, max_time=args.max_step*args.stride)

agents = []
for idx, inter_id in enumerate(world.inter_list):
    action_space = spaces.Discrete(8)
    agent = NonModularAgent(world, idx, 'observation', action_space)
    agents.append(agent)
agent_shared = NonModularAgent(world, -1, 'decision', spaces.Discrete(8))

env = TSCEnv(world, agents)

# train
def train(args, env):
    exp_num = 0
    total_mean_episode_rewards = []
    for episode in range(args.episode):
        computer_monitor()
        last_obs, last_states = env.reset()
        episode_rewards = [0 for _ in range(len(agents))]
        cur_step = 0
        while cur_step < args.max_step:
            actions = []
            for idx in range(len(agents)):
                if exp_num >= agent_shared.learning_start:
                    actions.append(agent_shared.choose_action(last_obs[idx], 'train'))
                else:
                    actions.append(agent_shared.sample())
    
            rewards_list = []
            obs, dones, states, infos = None, None, None, None
            for i in range(args.stride):
                if i == args.stride - 1:
                    env.stat_through_cnt()
                obs, rewards, dones, infos = env.step(actions)
                states = infos['states']
                rewards_list.append(rewards)
            cur_step += 1
            mean_rewards = np.mean(rewards_list, axis=0)

            for idx in range(len(agents)):
                agent_shared.remember(last_obs[idx], last_states[idx], actions[idx], mean_rewards[idx], states[idx])
                episode_rewards[idx] += mean_rewards[idx]
                exp_num += 1

            last_obs = obs
            last_states = states

            if exp_num >= agent_shared.learning_start and cur_step % agent_shared.update_model_freq == 0:
                if args.mode == 'meta_train':
                    agent_shared.meta_replay()
                elif args.mode == 'finetune':
                    agent_shared.replay()
                else:
                    raise ValueError()
                
            if all(dones):
                break
        
        if episode % args.save_model_freq == args.save_model_freq - 1:
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)
            agent_shared.save_model(args.save_model_dir)
        
        total_mean_episode_reward = np.mean([(episode_reward / args.max_step) for episode_reward in episode_rewards])
        total_mean_episode_rewards.append(total_mean_episode_reward)
        print(f'episode {episode} rewards: {total_mean_episode_reward}')
        
def load_model(model_dir, isPart):
    if model_dir:
        _model = os.path.join(model_dir, 'non_modular_model.pth')
    else:
        _model = None
    global agent_shared
    if isPart:
        agent_shared.load_part_model(_model)
    else:
        agent_shared.load_model(_model)


if __name__ == '__main__':
    load_model(args.model_dir, args.isPart)
    train(args, env)
