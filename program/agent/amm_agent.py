import os
import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn, tensor, Tensor

from network import ObservationAdapter

# global variables
_state_dim = 60  # A single state's size is (8, _state_dim).
_phase_pass_veh_num = 20  # The number of vehicles that are thought that
                          # can pass an intersection in a traffic light phase.


class StateActionDynamicsModel(nn.Module):
    def __init__(self):
        super(StateActionDynamicsModel, self).__init__()
        self.relu = nn.ReLU()
        # state + action --> next_state
        self.state_action_fc1_1 = nn.Linear(8*_state_dim, 256)
        self.state_action_fc1_2 = nn.Linear(256, 128)
        
        self.state_action_fc2_1 = nn.Linear(8, 64)
        self.state_action_fc2_2 = nn.Linear(64, 64)
        
        self.state_action_fc3_1 = nn.Linear(192, 512)
        self.state_action_fc3_2 = nn.Linear(512, 8*_state_dim)
        
    def forward(self, x):
        x = x.view(-1, 8*_state_dim + 8)
        # state + action --> next_state
        x_1 = self.relu(self.state_action_fc1_1(x[:, :-8]))
        x_1 = self.state_action_fc1_2(x_1)
        
        x_2 = self.relu(self.state_action_fc2_1(x[:, -8:]))
        x_2 = self.state_action_fc2_2(x_2)
        
        x = self.relu(torch.cat((x_1, x_2), dim=1))
        x = self.relu(self.state_action_fc3_1(x))
        x = self.state_action_fc3_2(x)
        return x

class PlanningStateLoss(nn.Module):
    '''
    Calculate loss of input states and target states.
    
    Args:
        input: input states. It can be a batch with 
        several states or a signal state.
        
        target: target states. It must have the same 
        shape as input.
    
    Returns:
        Loss of input and target.'''
    def __init__(self):
        super(PlanningStateLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = input.view(-1, 8*_state_dim)
        target = target.view(-1, 8*_state_dim)
        assert input.shape == target.shape
        assert (input.shape)[1] == 8*_state_dim
        discount = 0.1
        loss = tensor(0, dtype=torch.float, requires_grad=True).to(self.device)
        for lane in range(8):
            weight = 1.0
            for i in range(int(_state_dim/_phase_pass_veh_num)):
                input_sum = torch.sum(input[:, lane*_state_dim + i*_phase_pass_veh_num : lane*_state_dim + (i+1)*_phase_pass_veh_num], dim=1)
                target_sum =  torch.sum(target[:, lane*_state_dim + i*_phase_pass_veh_num : lane*_state_dim + (i+1)*_phase_pass_veh_num], dim=1)
                loss = loss + weight * torch.mean(torch.pow((input_sum - target_sum), 2))
                weight *= discount
        return loss


class AMMAgent:
    def __init__(self, world, inter_idx, mode, action_space):
        self.world = world
        self.inter_idx = inter_idx
        self.mode = mode
        
        if self.mode == 'observation':
            pass
        elif self.mode == 'decision':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.action_space = action_space
            self.epsilon = 0.1  # exploration rate
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            
            self.replay_buffer = []
            self.batch_size = len(self.world.inter_list)
            if self.batch_size > 64:
                self.batch_size = 64
            if self.batch_size < 8:
                self.batch_size = 8
            self.learning_start = 0 * 180 * len(self.world.inter_list)
            self.max_memory_cap = 20 * 180 * len(self.world.inter_list)
            
            self.learning_rate = 2e-5
            self.update_model_freq = 1
            
            self.dynamics_model = StateActionDynamicsModel().to(self.device)
            self.dynamics_optim = torch.optim.Adam(self.dynamics_model.parameters(), lr=self.learning_rate)
            self.dynamics_criterion = PlanningStateLoss().to(self.device)

            self.observation_adapter = ObservationAdapter().to(self.device)
            self.adapter_optim = torch.optim.Adam(self.observation_adapter.parameters(), lr=self.learning_rate)
            self.adapter_criterion = PlanningStateLoss().to(self.device)
            
            self.action_matrix_mapping = {0: [0, 0, 0, 0, 0, 0, 0, 0],
                                          1: [0, 1, 0, 0, 0, 1, 0, 0],
                                          2: [1, 0, 0, 0, 1, 0, 0, 0],
                                          3: [0, 0, 0, 1, 0, 0, 0, 1],
                                          4: [0, 0, 1, 0, 0, 0, 1, 0],
                                          5: [1, 1, 0, 0, 0, 0, 0, 0],
                                          6: [0, 0, 1, 1, 0, 0, 0, 0],
                                          7: [0, 0, 0, 0, 1, 1, 0, 0],
                                          8: [0, 0, 0, 0, 0, 0, 1, 1]}
        else:
            raise ValueError(f'The argument of mode "{self.mode}" is invalid.')
    
    def get_ob(self):
        return self.world.get_ob(self.inter_idx)
    
    def get_state(self):
        return self.world.get_state(self.inter_idx)
    
    def get_reward(self):
        return self.world.get_reward(self.inter_idx)
    
    def is_done(self):
        return self.world.is_done(self.inter_idx)
    
    def sample(self):
        return self.action_space.sample()

    def choose_action(self, ob: np.ndarray, mode: str) -> int:
        if mode == 'train':
            if random.random() <= self.epsilon:
                return self.action_space.sample()
            else:
                with torch.no_grad():
                    ob_tensor = torch.from_numpy(ob).float()
                    state = self.eval_ob_to_state(ob_tensor)
                    action = self.eval_state_to_action(state)
                return action
        elif mode == 'eval':
            with torch.no_grad():
                ob_tensor = torch.from_numpy(ob).float()
                state = self.eval_ob_to_state(ob_tensor)
                action = self.eval_state_to_action(state)
            return action
        else:
            raise ValueError()
                
    
    def eval_ob_to_state(self, ob: Tensor) -> Tensor:
        '''
        An observation to a state in evaluation mode.
        
        Args:
            ob: An observation. Tensor dtype: torch.float.
            
        Returns:
            A state.
        '''
        self.observation_adapter.eval()
        state = self.observation_adapter(ob.to(self.device))
        return state.cpu().view(8, _state_dim)
    
    def eval_state_to_action(self, state: Tensor) -> int:
        '''
        A state to an action in evaluation mode.
        
        Args:
            state: A state. Tensor dtype: torch.float.
            
        Returns:
            An action.
        '''
        self.dynamics_model.eval()
        state_action = []
        for i in range(self.action_space.n):
            action = tensor(self.action_matrix_mapping[i+1], dtype=torch.float).unsqueeze(dim=1)
            state_action.append(torch.cat((state, action), dim=1))
        state_action = torch.stack(state_action, dim=0)
        next_states = self.dynamics_model(state_action.to(self.device))
        rewards = self.states_to_rewards(next_states)
        max_reward = torch.max(rewards).item()
        idx = torch.nonzero(rewards==max_reward).squeeze(dim=1)
        return idx[random.randint(0, len(idx)-1)].item()
    
    def states_to_rewards(self, states: Tensor) -> Tensor:
        '''
        Next states to rewards.
        
        Args:
            states: state_{t+1}s.
            
        Returns:
            rewards: negative values. Greater is better.
        '''
        rewards = []
        for state in states:
            reward = 0.
            for i in range(8):
                reward += torch.sum(state[i*_state_dim : i*_state_dim + _phase_pass_veh_num]).item()
            rewards.append(reward)
        rewards = -1 * tensor(rewards, dtype=torch.float)
        return rewards
    
    def remember(self, ob, state, action, reward, next_state):
        if len(self.replay_buffer) >= self.max_memory_cap:
            del self.replay_buffer[0]
        if len(self.replay_buffer) > self.max_memory_cap:
            raise RuntimeError('The number of experiences is greater than capacity of replay buffer.')
        self.replay_buffer.append([ob, state, action, reward, next_state])
        
    def replay(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs = []
        states = []
        actions = []
        rewards = []
        next_states = []
        with torch.no_grad():
            for data in batch:
                obs.append(tensor(data[0], dtype=torch.float))
                states.append(tensor(data[1], dtype=torch.float))
                actions.append(data[2])
                rewards.append(data[3])
                next_states.append(tensor(data[4], dtype=torch.float))
            obs = torch.stack(obs, dim=0)
            states = torch.stack(states, dim=0)
            actions = tensor(actions, dtype=torch.int64)
            rewards = tensor(rewards, dtype=torch.float)
            next_states = torch.stack(next_states, dim=0)

            state_action = []
            for i in range(self.batch_size):
                action = tensor(self.action_matrix_mapping[actions[i].item()+1], dtype=torch.float).unsqueeze(dim=1)
                state_action.append(torch.cat((states[i], action), dim=1))
            state_action = torch.stack(state_action, dim=0)

        self.observation_adapter.train()
        adapter_pred = self.observation_adapter(obs.to(self.device))
        adapter_loss = self.adapter_criterion(adapter_pred.to(self.device), states.to(self.device))
        self.adapter_optim.zero_grad()
        adapter_loss.backward()
        self.adapter_optim.step()
        print(f'adapter loss: {adapter_loss.item()}')

        self.dynamics_model.train()
        dynamics_pred = self.dynamics_model(state_action.to(self.device))
        dynamics_loss = self.dynamics_criterion(dynamics_pred.to(self.device), next_states.to(self.device))
        self.dynamics_optim.zero_grad()
        dynamics_loss.backward()
        self.dynamics_optim.step()
        print(f'dynamics loss: {dynamics_loss.item()}')
        
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
    
    def save_model(self, save_dir):
        t = time.localtime()
        path = os.path.join(save_dir, f'AMMAgent_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}')
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.dynamics_model.state_dict(), os.path.join(path, 'dynamics_model.pth'))
        torch.save(self.observation_adapter.state_dict(), os.path.join(path, 'observation_adapter.pth'))
    
    def load_model(self, adapter_model_file=None, dynamics_model_file=None):
        if adapter_model_file:
            state_dict = torch.load(adapter_model_file, map_location=self.device)
            self.observation_adapter.load_state_dict(state_dict)
        if dynamics_model_file:
            state_dict = torch.load(dynamics_model_file, map_location=self.device)
            self.dynamics_model.load_state_dict(state_dict)
        
