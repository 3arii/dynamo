import os, sys, time
import numpy as np
import scipy.io as spio
import torch
from iflow.dataset.generic_dataset import Dataset

directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..','data')) + '/LASA_dataset/'

class LASA():
    def __init__(self, filenames, device=torch.device('cpu')):

        ## Define Variables and Load trajectories ##
        self.filenames = filenames
        self.device = device
        self.trajs_real = []

        for filename in filenames:
            mat = spio.loadmat(directory + filename + '.mat', squeeze_me=True)
            for demo_i in mat['demos']:
                x = demo_i[0]
                y = demo_i[1]
                tr_i = np.stack((x, y))
                self.trajs_real.append(tr_i.T)

        if not self.trajs_real:
            raise ValueError("Trajectories data is empty")

        # Concatenating and processing trajectories
        trajs_np = np.concatenate(self.trajs_real, axis=0)
        self.trj_length = trajs_np.shape[0]
        self.n_dims = trajs_np.shape[1]

        ## Normalize trajectories
        self.mean = np.mean(trajs_np, axis=0)
        self.std = np.std(trajs_np, axis=0)
        trajs_normalized = (trajs_np - self.mean) / self.std
        
        ## Build Train Dataset
        self.train_data = [trajs_normalized]

        self.dataset = Dataset(trajs=self.train_data, device=device)

    def unormalize(self, Xn):
        X = Xn * self.std + self.mean
        return X

if __name__ == "__main__":
    filenames = ['GShape', 'Trapezoid']  # Example shapes
    device = torch.device('cpu')
    lasa = LASA(filenames, device)
    print(lasa)
