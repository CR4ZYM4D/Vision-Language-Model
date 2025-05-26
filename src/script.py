import torch
import torch.nn as nn

vector_dimension = 768 

linear_dimension = 3072 # dimension of the linear layer of feed forward for normalization

image_size = 224 # pixels in each dimension of image

num_channels = 3 # number of channels in image (R, G, B)

patch_size = 16 # number of patches/ blocks we break each image into

attention_heads = 12 # number of attention heads

num_layers = 12 # number of transformer block layers in the model

dropout = 0.4

normalization_constant = 1e-6 # proportionality constant for normalization layer

num_image_tokens : int = None # number of tokens produced for each image i.e. it produces a list of vectors/ embeddings for a patch of each image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)