import os
import sys
import argparse
from pathlib import Path
import time

from gymnasium import spaces

sys.path.append(str(Path(__file__).resolve().parents[1]))
from world import World
from env import TSCEnv
from agent.amm_agent import AMMAgent


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='Cityflow_run/cfg/config_28x7.json')
parser.add_argument('--thread_num', type=int, default=20)
parser.add_argument('--max_step', type=int, default=180)
parser.add_argument('--stride', type=int, default=20)
parser.add_argument('--adapter_model_dir', type=str, required=True)
parser.add_argument('--dynamics_model_dir', type=str, required=True)

args = parser.parse_args()

# build world
world = World(config_path=args.config, thread_num=args.thread_num, max_time=args.max_step*args.stride)

agents = []
for idx, inter_id in enumerate(world.inter_list):
    action_space = spaces.Discrete(8)
    agent = AMMAgent(world, idx, 'observation', action_space)
    agents.append(agent)
agent_shared = AMMAgent(world, -1, 'decision', spaces.Discrete(8))

env = TSCEnv(world, agents)

# test
def test(args, env):
    observation_adapter = os.path.join(args.adapter_model_dir, 'observation_adapter.pth')
    dynamics_model = os.path.join(args.dynamics_model_dir, 'dynamics_model.pth')
    agent_shared.load_model(observation_adapter, dynamics_model)
    print(f'params num: {sum([param.nelement() for param in agent_shared.dynamics_model.parameters()])}')

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
