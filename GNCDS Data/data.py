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


        self.pos = rearrange(self.pos, 'd c n -> (d n) c').float()
        self.vel = rearrange(self.vel, 'd c n -> (d n) c').float()

        
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

class NormalisedStackedLASADataPendulum(Dataset):
    def __init__(self, dataset_path):
        data = np.load(dataset_path)
        X = data['X']  # Assuming this is position data
        Y = data['Y']  # Assuming this represents velocity or other target data
        
        # Convert to torch tensors
        self.pos = torch.tensor(X, dtype=torch.float32)
        self.vel = torch.tensor(Y, dtype=torch.float32)
        
        # Normalize position data
        self.pos_mean = self.pos.mean(dim=0)
        self.pos_std = self.pos.std(dim=0)
        self.pos = (self.pos - self.pos_mean) / self.pos_std
        
        # Normalize velocity data similarly if applicable
        # Adapt if velocity has its own mean/std. Using pos mean/std for simplicity
        self.vel = (self.vel - self.pos_mean) / self.pos_std
        
        # Calculate equilibrium position (pos_eq) - example calculation
        # Here, simply taking the mean as a placeholder. Adjust based on actual requirements.
        self.pos_eq = self.pos_mean.unsqueeze(0)  # Ensure it is 2D for subscripting

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        return self.pos[idx], self.vel[idx]

    def standardize_X(self, X):
        return (X - self.X_mean) / self.X_std
    
    def standardize_Y(self, Y):
        return (Y - self.Y_mean) / self.Y_std
    
    def unstandardize_X(self, X):
        return X * self.X_std + self.X_mean
    
    def unstandardize_Y(self, Y):
        return Y * self.Y_std + self.Y_mean
    
class NormalisedStackedLASAData(StackedLASAData):
    def __init__(self, dataset_path):
        data = np.load(dataset_path)
        X = data['X']  # Position data
        Y = data['Y']  # Assuming this is velocity or some target

        # Convert to torch tensors
        self.pos = torch.tensor(X, dtype=torch.float32)
        self.vel = torch.tensor(Y, dtype=torch.float32)

        # Assume pos_eq needs to be defined; here we use a placeholder approach
        # This should be adapted based on how pos_eq is actually determined for your case
        if 'pos_eq' in data:
            self.pos_eq = torch.tensor(data['pos_eq'], dtype=torch.float32)
        else:
            # Placeholder if pos_eq is not directly available; adapt as necessary
            self.pos_eq = torch.zeros_like(self.pos[0:1])

        # Normalize position data
        self.pos_mean = self.pos.mean(dim=0)
        self.pos_std = self.pos.std(dim=0)
        self.pos = (self.pos - self.pos_mean) / self.pos_std

        # Normalize velocity data similarly if applicable
        self.vel = (self.vel - self.pos_mean) / self.pos_std  # Adapt if vel has its own mean/std

        # Normalize equilibrium position
        self.pos_eq = (self.pos_eq - self.pos_mean) / self.pos_std

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        return self.pos[idx], self.vel[idx]

    def standardize_pos(self, pos):
        return (pos - self.pos_mean) / self.pos_std
    
    def standardize_vel(self, vel):
        return vel / self.vel_std
    
    def unstandardize_pos(self, pos):
        return pos * self.pos_std + self.pos_mean
    
    def unstandardize_vel(self, vel):
        return vel * self.vel_std 
    
class DiscreteNormalisedLASAData(NormalisedStackedLASAData):
    def __init__(self, dataset_names, start = 15, num_demos = None):
        super().__init__(dataset_names=dataset_names, start=start, num_demos=num_demos)
    
    def __getitem__(self, idx):
        if idx % self.traj_len == self.traj_len - 1:
            return self.pos[idx], self.pos[idx]
        return self.pos[idx], self.pos[idx + 1] 


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




class DiscreteLASAData(Dataset):
    def __init__(self, dataset_name, start = 15, num_demos = None):
        super().__init__()
        self.pos, self.vel = lasa_to_torch(dataset_name, start)
        if num_demos is not None and 0 < num_demos < self.pos.shape[0]:
            self.pos = self.pos[0:num_demos]

        self.dataset_name = dataset_name
        self.start = start

        # pos is (num_demos, 2, num_points)
        # vel is (num_demos, 2, num_points)

        self.traj_len = self.pos.shape[2]

        self.pos_eq = self.pos[:, :, -1]
        
        self.next_pos = self.pos[:, :, 1:]
        # add pos_eq to the end of next_pos
        self.next_pos = torch.cat([self.next_pos, self.pos_eq.unsqueeze(2)], dim=2)
        self.pos = rearrange(self.pos, 'd c n -> (d n) c')
        self.next_pos = rearrange(self.next_pos, 'd c n -> (d n) c')

    def __len__(self):
        return self.pos.shape[0]
    
    def __getitem__(self, idx):
        return self.pos[idx], self.next_pos[idx]
    
class DiscreteStackedLASAData(Dataset):
    def __init__(self, dataset_names, start = 15, num_demos = None):
        super().__init__()
        datasets = []
        for dataset_name in dataset_names:
            dataset = DiscreteLASAData(dataset_name, start=start, num_demos=num_demos)
            datasets.append(dataset)
        
        self.traj_len  = datasets[0].traj_len # should be the same for all 
        
        
        # stack pos data
        self.pos = torch.cat([dataset.pos for dataset in datasets], dim=1)
        # stack vel data
        self.next_pos = torch.cat([dataset.next_pos for dataset in datasets], dim=1)
        #stack eq data
        self.pos_eq = torch.cat([dataset.pos_eq for dataset in datasets], dim=1)

    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, idx):
        return self.pos[idx], self.next_pos[idx]
    
class DiscreteNormalisedStackedLASAData(DiscreteStackedLASAData):
    def __init__(self, dataset_names, start=15, num_demos = None):
        super().__init__(dataset_names, start, num_demos)
        
        # normalise pos data
        self.pos_mean = self.pos.mean(dim=0)
        self.pos_std = self.pos.std(dim=0)
        # assert pos_mean is of size [len(dataset_names) * 2]
        assert self.pos_mean.shape[0] == len(dataset_names) * 2


        self.pos = (self.pos - self.pos_mean) / self.pos_std

        self.next_pos = (self.next_pos - self.pos_mean) / self.pos_std

        self.pos_eq = (self.pos_eq - self.pos_mean) / self.pos_std
        
    
    def standardize_pos(self, pos):
        return (pos - self.pos_mean) / self.pos_std

    def unstandardize_pos(self, pos):
        return pos * self.pos_std + self.pos_mean



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
