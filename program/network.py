import torch
import torch.nn as nn
from torch import Tensor, tensor


_ob_dim = 3  # A single observation's size is (8, _ob_dim).
_state_dim = 60
_phase_pass_veh_num = 20


class ObservationAdapter(nn.Module):
    def __init__(self):
        super(ObservationAdapter, self).__init__()
        self.relu = nn.ReLU()
        # observation --> state
        self.adapter_fc1 = nn.Linear(8*_ob_dim, 1024)
        self.adapter_fc2 = nn.Linear(1024, 1024)
        self.adapter_fc3 = nn.Linear(1024, 8*_state_dim)
    
    def forward(self, x):
        x = x.view(-1, 8*_ob_dim) 
        # observation --> state
        x = self.relu(self.adapter_fc1(x))
        x = self.relu(self.adapter_fc2(x))
        x = self.adapter_fc3(x)
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
    
    
