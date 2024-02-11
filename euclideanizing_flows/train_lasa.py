from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from euclideanizing_flows.flows import BijectionNet, NaturalGradientDescentVelNet
from euclideanizing_flows.train_utils import train
from euclideanizing_flows.plot_utils import *
from euclideanizing_flows.data_utils import LASA
from euclideanizing_flows.concat_dataset import LASAConcatenated
import argparse

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def dtwd(traj1, traj2):
    dtwd_value = 0
    count = 0
    for i in traj1:
        dtwd_value += min(euclidean_distance(i, j) for j in traj2)
        count += 1
    for j in traj2:
        dtwd_value += min(euclidean_distance(i, j) for i in traj1)
        count += 1
    normalized_dtwd = dtwd_value / count
    return normalized_dtwd

parser = argparse.ArgumentParser(description='Euclideanizing flows for learning stable dynamical systems')
parser.add_argument('--data-name', type=str, default='GShape', help='name of the letter in LASA dataset')
args = parser.parse_args()

data_name = args.data_name
data_names = ['Trapezoid', 'GShape']
test_learner_model = True
load_learner_model = False
coupling_network_type = 'rffn'
plot_resolution = 0.01

if coupling_network_type == 'fcnn':
    num_blocks = 7
    num_hidden = 100
    t_act = 'elu'
    s_act = 'elu'
    minibatch_mode = True
    batch_size = 64
    learning_rate = 0.0005
    sigma = None
    print('WARNING: FCNN params are not tuned!! ')
elif coupling_network_type == 'rffn':
    num_blocks = 10
    num_hidden = 200
    sigma = 0.45
    minibatch_mode = False
    batch_size = 64
    s_act = None
    t_act = None
    learning_rate = 0.0001
else:
    raise TypeError('Coupling layer network not defined!')

eps = 1e-12
no_cuda = True
seed = None
weight_regularizer = 1e-10
epochs = 3
loss_clip = 1e3
clip_gradient = True
clip_value_grad = 0.1
log_freq = 10
plot_freq = 200
stopping_thresh = 250

cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

print('Loading dataset...')
# dataset = LASA(data_name = data_name) for the normal LASA dataset
dataset = LASAConcatenated(data_names=data_names)
goal = dataset.goal
idx = dataset.idx

x_train = dataset.x
xd_train = dataset.xd

# Normalize datasets
trajs_np = np.concatenate((x_train, xd_train), axis=0)
mean = np.mean(trajs_np, axis=0)
std = np.std(trajs_np, axis=0, ddof=1)  # Use ddof=1 for sample standard deviation
normalize_ = lambda x: (x - mean) / std

n_dims = dataset.n_dims
n_pts = dataset.n_pts
dt = dataset.dt

dataset_list = []
time_list = []
expert_traj_list = []
s0_list = []
dtwd_values = []
learner_traj_list = []
t_final_list = []
for n in range(len(idx) - 1):
    x_traj = normalize_(x_train[idx[n]:idx[n + 1]])
    xd_traj = normalize_(xd_train[idx[n]:idx[n + 1]])
    x_traj_tensor = torch.from_numpy(x_traj).float()
    xd_traj_tensor = torch.from_numpy(xd_traj).float()
    s0_list.append(x_traj_tensor[0].numpy())
    traj_dataset = TensorDataset(x_traj_tensor, xd_traj_tensor)
    expert_traj_list.append(x_traj_tensor)
    dataset_list.append(traj_dataset)
    t_final = dt * (x_traj_tensor.shape[0] - 1)
    t_final_list.append(t_final)
    t_eval = np.arange(0., t_final + dt, dt)
    time_list.append(t_eval)

n_experts = len(dataset_list)
x_train_normalized = normalize_(x_train)
xd_train_normalized = normalize_(xd_train)
x_train_tensor = torch.from_numpy(x_train_normalized).float()
xd_train_tensor = torch.from_numpy(xd_train_normalized).float()

if not minibatch_mode:
    batch_size = xd_train_tensor.shape[0]

xmin, xmax = np.min(x_train_normalized[:, 0]), np.max(x_train_normalized[:, 0])
ymin, ymax = np.min(x_train_normalized[:, 1]), np.max(x_train_normalized[:, 1])
x_lim = [[xmin - 0.1, xmax + 0.1], [ymin - 0.1, ymax + 0.1]]

taskmap_net = BijectionNet(num_dims=n_dims, num_blocks=num_blocks, num_hidden=num_hidden, s_act=s_act, t_act=t_act,
                           sigma=sigma, coupling_network_type=coupling_network_type).to(device)

y_pot_grad_fcn = lambda y: torch.nn.functional.normalize(y, dim=1)

euclideanization_net = NaturalGradientDescentVelNet(taskmap_fcn=taskmap_net,
                                                    grad_potential_fcn=y_pot_grad_fcn,
                                                    origin=torch.from_numpy(normalize_(goal)).float().to(device),
                                                    scale_vel=True,
                                                    is_diffeomorphism=True,
                                                    n_dim_x=n_dims,
                                                    n_dim_y=n_dims,
                                                    eps=eps,
                                                    device=device).to(device)

learner_model = euclideanization_net

if not load_learner_model:
    print('Training model ...')
    optimizer = optim.Adam(learner_model.parameters(), lr=learning_rate, weight_decay=weight_regularizer)
    criterion = nn.SmoothL1Loss()
    loss_fn = criterion

    dataset = TensorDataset(x_train_tensor, xd_train_tensor)
    learner_model.train()
    best_model, train_loss = train(learner_model, loss_fn, optimizer, dataset, epochs, batch_size, stopping_thresh)

    print('Training loss: {:.4f}'.format(train_loss))

    try:
        os.makedirs('models')
    except FileExistsError:
        pass  # Directory already exists, no need to raise an error

    torch.save(learner_model.state_dict(), os.path.join('models', '{}.pt'.format(data_name)))

else:
    print('Loading model ...')
    learner_model.load_state_dict(torch.load(os.path.join('models', '{}.pt'.format(data_name)), map_location=device))

if test_learner_model:
    # Omitted plotting and evaluation code for brevity
    pass
 
if len(expert_traj_list) >= 2:
    # Convert the tensors to numpy arrays if they are not already

    for n in range(n_experts):
        s0 = s0_list[n]
        t_final = t_final_list[n]
        learner_traj = generate_trajectories(learner_model, s0, order=1, return_label=False, t_step=dt, t_final=t_final,
                                             method='euler')

        learner_traj_list.append(learner_traj)

    learner_traj = generate_trajectories(learner_model, s0, order=1, return_label=False, t_step=dt, t_final=t_final,
                                             method='euler')

    for i in range(3):
        traj1 = learner_traj_list[i].detach().cpu().numpy()
        traj2 = expert_traj_list[i].detach().cpu().numpy()

        # Compute the DTWD value
        dtwd_value = dtwd(traj1, traj2)

        print(dtwd_value)

        dtwd_values.append(dtwd_value)


    # Print the DTWD value
    print(f"DTWD between the first two trajectories: {dtwd_values}")
    print(f"Mean: {np.mean(dtwd_values)}")
else:
    print("Not enough trajectories for DTWD computation.")

