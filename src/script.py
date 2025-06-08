import torch
import torch.nn as nn

#moving config data to VisonModel,py

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)