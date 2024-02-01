import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np
from iflow.dataset import lasa_dataset
from iflow.dataset import concat_dataset
from torch.utils.data import DataLoader
from iflow import model
from iflow.trainers import goto_dynamics_train
from iflow.utils import to_torch
from iflow.visualization import visualize_latent_distribution, visualize_vector_field, visualize_2d_generated_trj
from iflow.test_measures import log_likelihood, iros_evaluation

# DTWD calculation function
def euclidean_distance(point1, point2):
    return np.sqrt(sum((np.array(point1) - np.array(point2)) ** 2))

def dtwd(traj1, traj2):
    dtwd_value = 0
    count = 0
    # Calculate the first summation term
    for i in traj1:
        dtwd_value += min(euclidean_distance(i, j) for j in traj2)
        count += 1  
    # Calculate the second summation term
    for j in traj2:
        dtwd_value += min(euclidean_distance(i, j) for i in traj1)
        count += 1
    # Normalize by the total number of points considered
    normalized_dtwd = dtwd_value / count
    #TODO: 
    return normalized_dtwd

percentage = .99
batch_size = 100
depth = 10
lr = 0.001
weight_decay = 0.
nr_epochs = 1000
filename = 'Gshape'
filenames = ["Gshape", "Trapezoid"]
device = torch.device('cpu')

def main_layer(dim):
    return model.CouplingLayer(dim)

def create_flow_seq(dim, depth):
    chain = []
    for i in range(depth):
        chain.append(main_layer(dim))
        chain.append(model.RandomPermutation(dim))
        chain.append(model.LULinear(dim))
    chain.append(main_layer(dim))
    return model.SequentialFlow(chain)

if __name__ == '__main__':
    data = concat_dataset.CLASA(filenames=filenames)
    dim = data.dim
    params = {'batch_size': batch_size, 'shuffle': True}
    dataloader = DataLoader(data.dataset, **params)
    
    dynamics = model.TanhStochasticDynamics(dim, dt=0.01, T_to_stable=2.5)
    flow = create_flow_seq(dim, depth)
    iflow = model.ContinuousDynamicFlow(dynamics=dynamics, model=flow, dim=dim).to(device)
    params = list(flow.parameters()) + list(dynamics.parameters())
    optimizer = optim.Adamax(params, lr=lr, weight_decay=weight_decay)
    
    with open('data_DTWD.txt', 'w') as f_dtwd:
        for epoch in range(nr_epochs):
            for local_x, local_y in dataloader:
                dataloader.dataset.set_step()
                optimizer.zero_grad()
                loss = goto_dynamics_train(iflow, local_x, local_y)
                loss.backward(retain_graph=True)
                optimizer.step()

            if epoch % 10 == 0:
                with torch.no_grad():
                    iflow.eval()

                    # Visualization and evaluation
                    # visualize_2d_generated_trj(data.train_data, iflow, device, fig_number=2)
                    # visualize_latent_distribution(data.train_data, iflow, device, fig_number=1)
                    # visualize_vector_field(data.train_data, iflow, device, fig_number=3)
                    # iros_evaluation(data.train_data, iflow, device)

                    # Calculate DTWD
                    traj1 = iflow.generate_trj(torch.Tensor(data.train_data[0][0]).unsqueeze(0), T=150) 
                    traj2 = data.train_data[0]
                    # print(len(data.train_data[0]))
                    # print(len(data.train_data[0][0]))
                    # print(type(data.train_data[0]))
                    # break
                    normalized_dtwd_value = dtwd(traj1, traj2)
                    f_dtwd.write(f"Epoch {epoch}: Normalized DTWD = {normalized_dtwd_value}\n")
                    print(f'Epoch {epoch}: Normalized DTWD = {normalized_dtwd_value}')

                    # Other metrics
                    step = 20
                    trj = data.train_data[0]
                    trj_x0 = to_torch(trj[:-step, :], device)
                    trj_x1 = to_torch(trj[step:, :], device)
                    ll = log_likelihood(trj_x0, trj_x1, step, iflow, device)
                    print(f'Epoch {epoch}: Log Likelihood = {ll}')
                    print(f'Epoch {epoch}: Variance of the latent dynamics = {torch.exp(iflow.dynamics.log_var).tolist()}')
                    print(f'Epoch {epoch}: Velocity of the latent dynamics = {iflow.dynamics.Kv[0, 0].item()}')
