import os
import sys
import tomllib as toml
# from .. import ROOTPATH

ROOTPATH = "/Users/ian/1_Projects/hw/bioe145/final"

def get_config(filename):
    filename = os.path.join(ROOTPATH, "configs", filename)

    with open(filename, 'rb') as fp:
        config = toml.load(fp)
    return config