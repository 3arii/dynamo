import torch
import numpy as np

# import torch dataset, loader
from torch.utils.data import Dataset, DataLoader

from einops import rearrange

import pyLasaDataset as lasa


data_set_names = ['Angle',
        'BendedLine',
        'CShape',
        'DoubleBendedLine',
        'GShape',
        'JShape',
        'JShape_2',
        'Khamesh',
        'LShape',
        'Leaf_1',
        'Leaf_2',
        'Line',
        'Multi_Models_1',
        'Multi_Models_2',
        'Multi_Models_3',
        'Multi_Models_4',
        'NShape',
        'PShape',
        'RShape',
        'Saeghe',
        'Sharpc',
        'Sine',
        'Snake',
        'Spoon',
        'Sshape',
        'Trapezoid',
        'WShape',
        'Worm',
        'Zshape',
        'heee']


def lasa_to_torch(dataset_name, start = 15):
    assert dataset_name in data_set_names
    lasa_data = getattr(lasa.DataSet, dataset_name)
    demos = lasa_data.demos
    pos = torch.tensor(np.array([demo.pos for demo in demos]))[:, :, start:]
    vel = torch.tensor(np.array([demo.vel for demo in demos]))[:, :, start:]
    return pos, vel

class LASAData(Dataset):
    def __init__(self, dataset_name, start = 15, num_demos = None):
        super().__init__()
        self.pos, self.vel = lasa_to_torch(dataset_name, start)
        if num_demos is not None and 0 < num_demos < self.pos.shape[0]:
            self.pos = self.pos[0:num_demos]
            self.vel = self.vel[0:num_demos]
        self.dataset_name = dataset_name
        self.start = start

        # pos is (num_demos, 2, num_points)
        # vel is (num_demos, 2, num_points)

        self.traj_len = self.pos.shape[2]

        self.pos_eq = self.pos[:, :, -1]
        self.vel_eq = self.vel[:, :, -1]


        self.pos = rearrange(self.pos, 'd c n -> (d n) c')
        self.vel = rearrange(self.vel, 'd c n -> (d n) c')

        
    def __len__(self):
        return self.pos.shape[0]
    
    def __getitem__(self, idx):
        demo_idx = idx // self.traj_len
        return self.pos[idx], self.vel[idx], self.pos_eq[demo_idx], self.vel_eq[demo_idx]
    
class StackedLASAData(Dataset):
    def __init__(self, dataset_names,  start = 15, num_demos = None):
        super().__init__()
        datasets = []
        for dataset_name in dataset_names:
            dataset = LASAData(dataset_name, start=start, num_demos=num_demos)
            datasets.append(dataset)
        
        self.traj_len  = datasets[0].traj_len # should be the same for all 
        
        # stack pos data
        self.pos = torch.cat([dataset.pos for dataset in datasets], dim=1)
        # stack vel data
        self.vel = torch.cat([dataset.vel for dataset in datasets], dim=1)
        #stack eq data
        self.pos_eq = torch.cat([dataset.pos_eq for dataset in datasets], dim=1)
        self.vel_eq = torch.cat([dataset.vel_eq for dataset in datasets], dim=1)
    
        
    def __len__(self):
        return self.pos.shape[0]
    
    def __getitem__(self, idx):
        demo_idx = idx // self.traj_len
        return self.pos[idx], self.vel[idx], self.pos_eq[demo_idx], self.vel_eq[demo_idx]

class LASAData_ic(Dataset):
    # Like LASAData, but values 2 and 3 are initial conditions rather than equilibrium points
    def __init__(self, dataset_name, start = 15, num_demos = None):
        super().__init__()
        self.pos, self.vel = lasa_to_torch(dataset_name, start)
        if num_demos is not None and 0 < num_demos < self.pos.shape[0]:
            self.pos = self.pos[0:num_demos]
            self.vel = self.vel[0:num_demos]
        self.dataset_name = dataset_name
        self.start = start

        # pos is (num_demos, 2, num_points)
        # vel is (num_demos, 2, num_points)

        self.traj_len = self.pos.shape[2]

        self.pos_0 = self.pos[:, :, 0]
        self.vel_0 = self.vel[:, :, 0]


        self.pos = rearrange(self.pos, 'd c n -> (d n) c')
        self.vel = rearrange(self.vel, 'd c n -> (d n) c')

        
    def __len__(self):
        return self.pos.shape[0]
    
    def __getitem__(self, idx):
        demo_idx = idx // self.traj_len
        return self.pos[idx], self.vel[idx], self.pos_0[demo_idx], self.vel_0[demo_idx]
    
class StackedLASAData_ic(Dataset):
    def __init__(self, dataset_names, start=15, num_demos = None):
        super().__init__()
        datasets = []
        for dataset_name in dataset_names:
            dataset = LASAData(dataset_name, start=start, num_demos=num_demos)
            datasets.append(dataset)
        self.traj_len  = datasets[0].traj_len
        
        # stack pos data
        self.pos = torch.cat([dataset.pos for dataset in datasets], dim=1)
        # stack vel data
        self.vel = torch.cat([dataset.vel for dataset in datasets], dim=1)
        #stack eq data
        self.pos_0 = torch.cat([dataset.pos_0 for dataset in datasets], dim=1)
        self.vel_0 = torch.cat([dataset.vel_0 for dataset in datasets], dim=1)

        self.traj_len = self.pos.shape[2]

    def __len__(self):
        return self.pos.shape[0]
    
    def __getitem__(self, idx):
        demo_idx = idx // self.traj_len
        return self.pos[idx], self.vel[idx], self.pos_0[demo_idx], self.vel_0[demo_idx]

def get_LASA_dataloader(dataset_name, batch_size=1, start = 15, shuffle = True, num_demos = None, ic=False):
    if ic:
        dataset = LASAData_ic(dataset_name, start, num_demos)
    else:
        dataset = LASAData(dataset_name, start, num_demos)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    return data_loader


def get_stacked_LASA_dataloader(dataset_names, batch_size=1, start = 15, shuffle = True, num_demos = None, ic=False):
    
    if ic:
        dataset = StackedLASAData_ic(dataset_names, start, num_demos)
    else:
        dataset = StackedLASAData(dataset_names, start, num_demos)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    return data_loader