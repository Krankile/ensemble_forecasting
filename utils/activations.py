from torch import nn


activations = {
    "relu": nn.ReLU, 
    "elu": nn.ELU, 
    "leaky": nn.LeakyReLU, 
}