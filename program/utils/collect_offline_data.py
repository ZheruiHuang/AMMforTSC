import os
import sys
import time
import random
import argparse
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from world import World

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='Cityflow_run/cfg/config_16x3.json')
parser.add_argument('--thread_num', type=int, default=20)
parser.add_argument('--max_step', type=int, default=180)
parser.add_argument('--action_interval', type=int, default=20)
parser.add_argument('--max_episode', type=int, default=10)
parser.add_argument('--random_action_ratio', type=float, default=1.0)
parser.add_argument('--output_dir', type=str, default='outputs/generated_data/16x3/')


class DataGenerator:
    def __init__(self, config_file, thread_num=1, max_step=180, 
                 action_interval=20, output_path='outputs/generated_data/'):
        self.max_step = max_step
        self.action_interval = action_interval
        self.output_path = output_path
        
        # Create engine
        self.world = World(config_path=config_file, 
                           thread_num=thread_num, 
                           max_time=max_step*action_interval)
        self.eng = self.world.eng 

        self.action_matrix_mapping = {0: [0, 0, 0, 0, 0, 0, 0, 0],
                                      1: [0, 1, 0, 0, 0, 1, 0, 0],
                                      2: [1, 0, 0, 0, 1, 0, 0, 0],
                                      3: [0, 0, 0, 1, 0, 0, 0, 1],
                                      4: [0, 0, 1, 0, 0, 0, 1, 0],
                                      5: [1, 1, 0, 0, 0, 0, 0, 0],
                                      6: [0, 0, 1, 1, 0, 0, 0, 0],
                                      7: [0, 0, 0, 0, 1, 1, 0, 0],
                                      8: [0, 0, 0, 0, 0, 0, 1, 1]}
        
       
            
    def generate_data(self, mode):
        inter_num = len(self.world.inter_list)      
        for step in range(self.max_step):
            obs = []
            states = []
            next_states = []
            actions = []
            rewards = [[] for _ in range(inter_num)]
            mean_rewards = []
            
            for inter_idx in range(inter_num):
                # obs
                obs.append(self.world.get_ob(inter_idx))
                
                # states
                states.append(self.world.get_state(inter_idx))
                
                # actions
                if mode == 0:
                    action = random.randint(0, 7)
                    actions.append(action)
                elif mode == 1:
                    if random.random() < 0.3:
                        action = random.randint(0, 7)
                    else:
                        action = ((step * self.action_interval) // 40) % 8
                    actions.append(action)
                else:
                    raise ValueError()
            
            for _ in range(self.action_interval):
                self.world.next_step(actions)
                for inter_idx in range(inter_num):
                    rewards[inter_idx].append(self.world.get_reward(inter_idx))
            for inter_idx in range(inter_num):
                mean_rewards.append(sum(rewards[inter_idx]) / len(rewards[inter_idx]))
            
            for inter_idx in range(inter_num):
                next_states.append(self.world.get_state(inter_idx))
            
            npz_dict = {'obs':np.array(obs), 
                        'states':np.array(states), 
                        'actions':np.array(actions), 
                        'next_states':np.array(next_states), 
                        'rewards':np.array(mean_rewards)}
            
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            np.savez(os.path.join(self.output_path, f'step{step}.npz'), **npz_dict)
    
    def reset(self):
        self.world.reset()

if __name__ == '__main__':
    args = parser.parse_args()
    generator = DataGenerator(config_file=args.config,
                              thread_num=args.thread_num,
                              max_step=args.max_step,
                              action_interval=args.action_interval)
    
    start_time = time.time()
    for episode in range(args.max_episode):
        generator.output_path = os.path.join(args.output_dir, f'episode{episode}/')
        generator.reset()
        print(f'Logging episode {episode} data...')
        if episode < args.max_episode * args.random_action_ratio:
            generator.generate_data(mode=0)
        else:
            generator.generate_data(mode=1)
    end_time = time.time()
    
    print(f'Done!\nMaxEpisode: {args.max_episode}; Runtime: {end_time - start_time:.2f}s')
