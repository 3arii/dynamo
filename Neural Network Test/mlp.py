from collections import OrderedDict
import torch.nn as nn

def get_training_model(inFeatures=4, hiddenDim=8, nbClasses=3):
    mlpModel = nn.Sequential(OrderedDict([
        ("hidden_layer_1", nn.Linear(inFeatures, hiddenDim)),
        ("activation_1", nn.ReLU()),
        ("output_layer", nn.Linear(hiddenDim, nbClasses))
    ]))

    return mlpModel