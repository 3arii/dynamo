import json
import argparse
from utils import *
from models import *
from data import *

import torch
import numpy as np

from tqdm import tqdm
import os

from m_flow.vector_transforms import create_vector_transform
from manifold_flow.transforms.projections import  CompositeProjection


# parse args
# parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str)

# args = parser.parse_args()

# # load config
# with open(args.config) as f:
#     config = json.load(f)

#     print("Config", args.config)
#     data_config = config['data']
#     model_config = config['model']
#     train_config = config['train']



dataset_names = ['Angle',
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
dataset_path = '/Users/deniz/Documents/Research/GNCDS/Pendulum Cache/p-3-test.npz'

# Initialize the dataset with the .npz file
dataset = NormalisedStackedLASADataPendulum(dataset_path)
train_loader = DataLoader(dataset, batch_size=200, shuffle=True)

# Training parameters
data_dim = dataset.pos.shape[1]  # Assuming X is 2D: [samples, features]
latent_dim = data_dim  # Adjust according to your model's architecture
flow_steps = 2  # Example flow steps

# Specify the save folder for models
save_folder = "Save Folder"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

epochs = 100  # Adjust according to your needs
lr = 0.001  # Learning rate

# Ensure the loss.txt file exists
loss_text_file = os.path.join(save_folder, 'loss.txt')
if not os.path.exists(loss_text_file):
    with open(loss_text_file, 'w') as f:
        f.write('')


def save_state(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, path)
    
def load_state(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] 
    return epoch


x_eq = dataset.pos_eq[0:1].float()
rq_transform = create_vector_transform( data_dim,
    flow_steps,
    linear_transform_type="lu",
    base_transform_type="rq-coupling",
    hidden_features=30,
    num_transform_blocks=2,
    dropout_probability=0.0,
    use_batch_norm=False,
    num_bins=10,
    tail_bound=10,
    apply_unconditional_transform=False,
    context_features=None) 

transform_c = CompositeProjection(rq_transform, data_dim, latent_dim)
transform = transform_c # change this
gncds = GNCDS(d=latent_dim, x_eq = None, hidden_dim = 16)
model_t = GNCDS_Transfrom(x_eq, transform=transform, model=gncds)
optimizer = torch.optim.Adam(model_t.parameters(), lr=lr)
epoch = 0
    
losses = []

for e in range(epoch+1, epochs + 1):
    total_loss = 0

    for i, (x, x_dot) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # Your model's forward pass to compute predictions
        x_dot_pred = model_t(x)
        
        # Compute loss
        loss = torch.nn.MSELoss()(x_dot_pred, x_dot)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch: ", e, "Loss: ", total_loss / len(train_loader))
    #write loss to file
    with open(loss_text_file, 'a') as f:
        f.write(f'{e}, {total_loss/ len(train_loader)}\n')

    # delete every model in save folder
    for f in os.listdir(save_folder):
        if f.endswith('.pt'):
            os.remove(os.path.join(save_folder, f))

    save_path = os.path.join(save_folder, f'model_e_{e}_loss_{loss:6.4f}.pt')
    save_state(model_t, optimizer, e, loss, save_path)
    losses.append(total_loss)
