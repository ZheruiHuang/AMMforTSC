import os
import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn, tensor, Tensor


# global variables
_ob_dim = 3
_state_dim = 60  # A single state's size is (8, _state_dim).
_phase_pass_veh_num = 20  # The number of vehicles that are thought that
                          # can pass an intersection in a traffic light phase.


class NonModularNet(nn.Module):
    def __init__(self):
        super(NonModularNet, self).__init__()
        self.relu = nn.ReLU()
        # observation --> state
        self.adapter_fc1 = nn.Linear(8*_ob_dim, 1024)
        self.adapter_fc2 = nn.Linear(1024, 1024)
        self.adapter_fc3 = nn.Linear(1024, 8*_state_dim + 8)
        
        # state + action --> next_state
        self.state_action_fc1_1 = nn.Linear(8*_state_dim, 1024)
        self.state_action_fc1_2 = nn.Linear(1024, 512)
        
        self.state_action_fc2_1 = nn.Linear(8, 512)
        self.state_action_fc2_2 = nn.Linear(512, 512)
        
        self.state_action_fc3_1 = nn.Linear(1024, 4096)
        self.state_action_fc3_2 = nn.Linear(4096, 8 * 8 * _state_dim)

    def forward(self, x):
        x = x.view(-1, 8*_ob_dim) 
        # observation --> state
        x = self.relu(self.adapter_fc1(x))
        x = self.relu(self.adapter_fc2(x))
        x = self.adapter_fc3(x)
        
        
        x = x.view(-1, 8*_state_dim+8)
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


class NonModularAgent:
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
            
            self.learning_rate = 1e-3
            self.update_model_freq = 1
            
            self.model = NonModularNet().to(self.device)
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.criterion = PlanningStateLoss().to(self.device)
            
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
                    action = self.eval_ob_to_action(ob_tensor)
                return action
        elif mode == 'eval':
            with torch.no_grad():
                ob_tensor = torch.from_numpy(ob).float()
                action = self.eval_ob_to_action(ob_tensor)
            return action
        else:
            raise ValueError()
                
    
    def eval_ob_to_action(self, ob: Tensor) -> Tensor:
        self.model.eval()
        next_states = self.model(ob.to(self.device)).view(8, 8*_state_dim)
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
        actions = []
        next_states = []
        with torch.no_grad():
            for data in batch:
                obs.append(tensor(data[0], dtype=torch.float))
                actions.append(data[2])
                next_states.append(tensor(data[4], dtype=torch.float))
            obs = torch.stack(obs, dim=0)
            actions = tensor(actions, dtype=torch.int64)
            next_states = torch.stack(next_states, dim=0)

        self.model.train()
        preds = self.model(obs.to(self.device))
        processed_preds = []
        for i, pred in enumerate(preds):
            pred = pred.view(8, 8*_state_dim)
            processed_preds.append(pred[actions[i]])
        processed_preds = torch.stack(processed_preds, dim=0)
        loss = self.criterion(processed_preds.to(self.device), next_states.to(self.device))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        print(f'loss: {loss.item()}')
    
    def meta_replay(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs = []
        actions = []
        next_states = []
        with torch.no_grad():
            for data in batch:
                obs.append(tensor(data[0], dtype=torch.float))
                actions.append(data[2])
                next_states.append(tensor(data[4], dtype=torch.float))
            obs = torch.stack(obs, dim=0)
            actions = tensor(actions, dtype=torch.int64)
            next_states = torch.stack(next_states, dim=0)

        self.model.train()
        params_dict_0 = self.model.state_dict()
        preds_0 = self.model(obs.to(self.device))
        processed_preds_0 = []
        for i, pred in enumerate(preds_0):
            pred = pred.view(8, 8*_state_dim)
            processed_preds_0.append(pred[actions[i]])
        processed_preds_0 = torch.stack(processed_preds_0, dim=0)
        loss = self.criterion(processed_preds_0.to(self.device), next_states.to(self.device))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        params_dict_1 = self.model.state_dict()
        preds_1 = self.model(obs.to(self.device))
        processed_preds_1 = []
        for i, pred in enumerate(preds_1):
            pred = pred.view(8, 8*_state_dim)
            processed_preds_1.append(pred[actions[i]])
        processed_preds_1 = torch.stack(processed_preds_1, dim=0)
        loss = self.criterion(processed_preds_1.to(self.device), next_states.to(self.device))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        params_dict_2 = self.model.state_dict()
        new_params = {}
        for param in params_dict_2:
            maml_grads = params_dict_2[param] - params_dict_1[param]
            new_params[param] = params_dict_0[param] + maml_grads
        
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def save_model(self, save_dir):
        t = time.localtime()
        path = os.path.join(save_dir, f'NonModularAgent_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}')
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), os.path.join(path, 'non_modular_model.pth'))
    
    def load_model(self, model_file):
        if model_file:
            state_dict = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(state_dict)
    
    def load_part_model(self, model_file):
        reserved_layers = ('state_action_fc1_1', 'state_action_fc1_2', 'state_action_fc2_1', 'state_action_fc2_2', 'state_action_fc3_1', 'state_action_fc3_2')
        attrs = ('weight', 'bias')
        reserved_params = []
        for layer in reserved_layers:
            for attr in attrs:
                reserved_params.append(f'{layer}.{attr}')
        if model_file:
            state_dict = torch.load(model_file, map_location=self.device)
            proc_dict = {}
            for key in state_dict:
                if key in reserved_params:
                    proc_dict[key] = state_dict[key]
                    print(f'load layer {key}')
                else:
                    print(f'NOT load layer {key}')
            self.model.load_state_dict(proc_dict, strict=False)
