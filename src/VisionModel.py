import torch
import torch.nn as nn
import torch.nn.functional as f 



# class to make the vision Transformer Model

class VisionTransformer(nn.Module):

    def __init__(self, vector_dimension, normalization_constant):
        super().__init__()

        self.embeddingLayer
        self.encoderLayer
        self.postNormalizationLayer = nn.LayerNorm(vector_dimension, eps = normalization_constant)

    def forward(self, image_tensor):

        hidden_state = self.embeddingLayer(image_tensor)

        hidden_state = self.encoderLayer(hidden_state)

        hidden_state = self.postNormalizationLayer(hidden_state)

        return hidden_state