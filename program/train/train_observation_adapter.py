# Train ObservationAdapter on offline dataset
import os
import sys
import time
import argparse
from pathlib import Path
from tqdm import tqdm

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from network import ObservationAdapter, PlanningStateLoss
from utils.dataset import build_data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='outputs/generated_data/28x7/')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--data_vol', type=int, default=2)
parser.add_argument('--save_dir', type=str, default='outputs/models/observation_adapter/28x7/')

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ObservationAdapter().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    criterion = PlanningStateLoss().to(device)
    data_loader = build_data_loader(data_dir=args.data_dir,
                                    vol=args.data_vol,
                                    shuffle=True, 
                                    )
    
    print('Training model...')
    for data in tqdm(data_loader):
        obs = data['obs'].to(device)
        states = data['states'].to(device)
        pred = net(obs)
        loss = criterion(pred, states)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    t = time.localtime()
    t_dir = os.path.join(args.save_dir, f'ObservationAdapter_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}')
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)
    path = os.path.join(t_dir, 'observation_adapter.pth')
    torch.save(net.state_dict(), path)


if __name__ == '__main__':
    train(parser.parse_args())
