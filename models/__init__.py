import os
import sys
import tomllib as toml
import torch
# from .. import ROOTPATH

ROOTPATH = "/Users/ian/1_Projects/hw/bioe145/final"

def save_config(config, run_name):
    filepath = os.path.join(ROOTPATH, "models", run_name)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    with open(filepath + "config.toml", 'w') as fp:
        toml.dump(config, fp)

def save_model(model_sd, encoder, decoder, optimizer_sd, run_name):
    filepath = os.path.join(ROOTPATH, "models", run_name)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    torch.save({"model": model_sd,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer_sd
               }, 
               filepath + "model.pt")