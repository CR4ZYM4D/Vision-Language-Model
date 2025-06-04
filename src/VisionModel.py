import torch
import torch.nn as nn
import torch.nn.functional as f 

# class for the embedding layer of the vision transformer model

class VisionEmbedding(nn.Module):

    def __init__(self, vector_dimension, patch_size, num_channels, image_size):
        super().__init__()

        self.convolutor = nn.Conv2d(
            in_channels = num_channels,  # 3 channels per image R, G, B
            out_channels= vector_dimension, 
            kernel_size= patch_size,
            stride = patch_size
            )
        
        self.num_patches = (image_size // patch_size) ** 2

        self.positionalEmbedding = nn.Embedding(self.num_patches, vector_dimension)

        self.register_buffer("positions", torch.arange(self.num_patches).expand((1, -1)), persistent= False)

    def forward(self, image_tensors):

        # image tensors are of shape [ batch_size x num_channels x height x width ] 

        # converting them to tensors of shape [ batch_size x vector_dimension x (height / patch_size) x (width / patch_size)]
        # using the CNN

        image_embeddings = self.convolutor(image_tensors) # [ batch_size x vector_dimension x (height / patch_size) x (width / patch_size) ]

        image_embeddings = image_embeddings.flatten(2) # [ batch_size x vector_dimension x num_patches ]

        image_embeddings = image_embeddings.transpose(1, 2) # [ batch_size x num_patches x vector_dimension ]

        return image_embeddings + self.positionalEmbedding(self.positions)

# class for a single encoder block of the vision transformer

class VisionEncoderBlock(nn.Module):

    def __init__(self, ):
        super().__init__(vector_dimension, normalization_constant, attention_heads, dropout, linear_dimension)

        self.layerNorm1 = nn.LayerNorm(vector_dimension, normalization_constant)
        self.layerNorm2 = nn.LayerNorm(vector_dimension, normalization_constant) 

        self.attentionBlock

        self.MLP 

    def forward(self, hidden_states):

        skip_connector = torch.clone(hidden_states)

        hidden_states = self.layerNorm1(hidden_states)

        hidden_states = self.attentionBlock()

        hidden_states = hidden_states + skip_connector

        skip_connector = hidden_states

        hidden_states = self.layerNorm2(hidden_states)

        return hidden_states + skip_connector

# class to make the vision Transformer Model

class VisionTransformer(nn.Module):

    def __init__(self, vector_dimension, normalization_constant, patch_size, num_channels, image_size):
        super().__init__()

        self.embeddingLayer = VisionEmbedding(vector_dimension, patch_size, num_channels, image_size)
        self.encoderLayer
        self.postNormalizationLayer = nn.LayerNorm(vector_dimension, eps = normalization_constant)

    def forward(self, image_tensors):

        hidden_state = self.embeddingLayer(image_tensors)

        hidden_state = self.encoderLayer(hidden_state)

        hidden_state = self.postNormalizationLayer(hidden_state)

        return hidden_state