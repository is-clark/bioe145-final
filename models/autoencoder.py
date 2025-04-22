import torch.nn as nn
import numpy as np

def find_hidden_dims(in_size, num_layers, out_size):
    jump = int(np.abs(np.ceil((in_size - out_size) / num_layers)))
    size = in_size
    layer_sizes = [in_size]
    if in_size > out_size:
        for _ in range(num_layers - 1):
            size -= jump
            layer_sizes.append(size)
    elif in_size < out_size:
        for _ in range(num_layers - 1):
            size += jump
            layer_sizes.append(size)
    else:
        layer_sizes = [out_size for _ in range(num_layers - 1)]
 
    return layer_sizes

class NodeLayer(nn.Sequential):
    def __init__(self, input_size, output_size):
        super(NodeLayer, self).__init__(
            nn.Linear(input_size, output_size),
            nn.ReLU()
        )

class MLP(nn.Module):
    def __init__(self, input_size, num_hidden_layers, output_size):
        super(MLP, self).__init__()

        hidden_dims = find_hidden_dims(input_size, num_hidden_layers, output_size)
        
        self.hidden_layers = nn.Sequential(*[NodeLayer(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims) - 2 + 1)])
        self.out_layer = nn.Linear(hidden_dims[-1], output_size)

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.out_layer(x)

class PBMCAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(PBMCAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
