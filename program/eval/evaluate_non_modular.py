import os
import sys
import time
import argparse
from pathlib import Path

from gymnasium import spaces

sys.path.append(str(Path(__file__).resolve().parents[1]))
from world import World
from env import TSCEnv
from agent.non_modular_agent import NonModularAgent


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='Cityflow_run/cfg/config_28x7.json')
parser.add_argument('--thread_num', type=int, default=20)
parser.add_argument('--max_step', type=int, default=180)
parser.add_argument('--stride', type=int, default=20)
parser.add_argument('--model_dir', type=str, required=True)

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

# test
def test(args, env):
    _model = os.path.join(args.model_dir, 'non_modular_model.pth')
    agent_shared.load_model(_model)

    # simulate
    obs, states = env.reset()
    actions = []
    travel_time_list = list()
    waiting_veh_num_list = list()
    for i in range(args.max_step):
        actions = []
        for agent_id, agent in enumerate(agents):
            actions.append(agent_shared.choose_action(obs[agent_id], 'eval'))
        for _ in range(args.stride):
            obs, rewards, dones, info = env.step(actions)
            travel_time_list.append(env.eng.get_average_travel_time())
            waiting_veh_cnt = list(env.eng.get_lane_waiting_vehicle_count().values())
            waiting_veh_num_list.append(sum(waiting_veh_cnt) / len(waiting_veh_cnt))
        print(env.eng.get_average_travel_time())
        print(i)

    print(f'Total average travel time: {sum(travel_time_list) / len(travel_time_list)}')
    print(f'Total average waiting vehicle number: {sum(waiting_veh_num_list) / len(waiting_veh_num_list)}')
    

if __name__ == '__main__':
    test(args, env)
    
