import yaml
import os
import torch

def create_folder(path):
    if not os.path.exists(path):
        # If it does not exist, create it
        os.makedirs(path)
        
        
        
