import os, sys, time
import numpy as np
import scipy.io as spio
import torch
from iflow.dataset.generic_dataset import Dataset

directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..','data')) + '/LASA_dataset/'

class CLASA():
    def __init__(self, filenames, device=torch.device('cpu')):

        ## Define Variables and Load trajectories ##
        self.filenames = filenames
        self.device = device
        self.dim = 2 * len(filenames)
        self.trajs_real = []

        for filename in filenames:
            trajs_real_f = []
            mat = spio.loadmat(directory + filename + '.mat', squeeze_me=True)
            for demo_i in mat['demos']:
                x = demo_i[0]
                y = demo_i[1]
                tr_i = np.stack((x, y))
                trajs_real_f.append(tr_i.T)

            # Concatenate all trajectories for a single filename
            traj_real_f = np.concatenate(trajs_real_f, axis=0)

            # Split into chunks of 150 points
            num_chunks = traj_real_f.shape[0] // 150
            chunks = [traj_real_f[i*150:(i+1)*150] for i in range(num_chunks)]
            self.trajs_real.append(chunks)

        # Concatenate corresponding chunks from each file
        concatenated_chunks = []
        for i in range(len(self.trajs_real[0])):
            chunk = np.concatenate([trajs_real[i] for trajs_real in self.trajs_real], axis=1)
            concatenated_chunks.append(chunk)

        # Normalize and finalize train_data
        self.train_data = []
        for traj in concatenated_chunks:
            traj_mean = np.mean(traj, axis=0)
            traj_std = np.std(traj, axis=0)
            traj_normalized = (traj - traj_mean) / traj_std
            self.train_data.append(traj_normalized)

        self.train_data = np.array(self.train_data)
        self.dataset = Dataset(trajs=self.train_data, device=device)

        print(self.train_data[0])
        print(len(self.train_data))

    def unormalize(self, Xn):
        X = Xn * self.std + self.mean
        return X

if __name__ == "__main__":
    filenames = ['GShape', 'Trapezoid']  # Example shapes
    device = torch.device('cpu')
    lasa = CLASA(filenames, device)
    print(lasa)