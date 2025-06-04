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


# class for the Vision Encoder block MLP layer

class VisionMLP(nn.Module):

    def __init__(self, vector_dimension, linear_dimension):
        super().__init__()

        self.layer1 = nn.Linear(vector_dimension, linear_dimension)
        self.layer2 = nn.Linear(linear_dimension, vector_dimension)

    def forward(self, hidden_state):

        hidden_state = self.layer1(hidden_state)

        hidden_state = f.gelu(hidden_state, approximate= "tanh")

        return self.layer2(hidden_state)
    
# class for the attention Block. Since it is an image transformer and not a langugae one, there is no need of a causal mask

class VisionAttention(nn.Module):

    def __init__(self, vector_dimension, attention_heads, dropout):
        super().__init__()

        self.num_heads = attention_heads
        self.dropout = nn.Dropout(dropout)
        self.vector_dimension = vector_dimension
        self.head_dimension = vector_dimension // attention_heads

        self.wQ = nn.Linear(vector_dimension, vector_dimension) # Query projection layer
        self.wK = nn.Linear(vector_dimension, vector_dimension) # Key projection layer
        self.wV = nn.Linear(vector_dimension, vector_dimension) # Value projection layer
        self.wO = nn.Linear(vector_dimension, vector_dimension) # Output projection layer

        self.scale = vector_dimension ** -0.5

    def forward(self, hidden_state):

        # hidden_state [ batch_size x num_patches x vector_dimension ]
        batch_size, sequence_len = hidden_state.size() # sequence len = num_patches

        queryProjection = self.wQ(hidden_state) # [ batch_size x num_patches x vector_dimension ]
        keyProjection = self.wQ(hidden_state) # [ batch_size x num_patches x vector_dimension ]
        valueProjection = self.wQ(hidden_state) # [ batch_size x num_patches x vector_dimension ]

        queryProjection = queryProjection.view(batch_size, sequence_len, self.num_heads, self.head_dimension) # [batch_size x sequence_len x num_heads x head_dimension]
        keyProjection = keyProjection.view(batch_size, sequence_len, self.num_heads, self.head_dimension) # [batch_size x sequence_len x num_heads x head_dimension]
        valueProjection = valueProjection.view(batch_size, sequence_len, self.num_heads, self.head_dimension) # [batch_size x sequence_len x num_heads x head_dimension]

        queryProjection = queryProjection.transpose(1, 2) # [batch_size x num_heads x sequence_len x head_dimension]
        keyProjection = keyProjection.transpose(1, 2) # [batch_size x num_heads x sequence_len x head_dimension]
        valueProjection = valueProjection.transpose(1, 2) # [batch_size x num_heads x sequence_len x head_dimension]

        attention_scores = torch.matmul(queryProjection, keyProjection.transpose(2, 3)) * self.scale # [batch_size x num_heads x sequence_len x sequence_len]

        attention_scores = f.softmax(attention_scores, dim = -1)

        attention_scores = torch.matmul(attention_scores, valueProjection) # [ batch_size x num_heads x  sequence_len x head_dimension]

        attention_scores = attention_scores.transpose(1, 2).contiguous() # [batch_size x sequence_len x num_heads x head_dimension]

        attention_scores = attention_scores.reshape(batch_size, sequence_len, self.vector_dimension) # [batch_size, sequence_len, vector_dimension]

        return self.wO(attention_scores)
    

# class for a single encoder block of the vision transformer

class VisionEncoderBlock(nn.Module):

    def __init__(self, vector_dimension, normalization_constant, attention_heads, dropout, linear_dimension):
        super().__init__()

        self.layerNorm1 = nn.LayerNorm(vector_dimension, normalization_constant)
        self.layerNorm2 = nn.LayerNorm(vector_dimension, normalization_constant) 

        self.attentionBlock = VisionAttention(vector_dimension, attention_heads, dropout)

        self.MLP = VisionMLP(vector_dimension, linear_dimension)

    def forward(self, hidden_states):

        skip_connector = torch.clone(hidden_states)

        hidden_states = self.layerNorm1(hidden_states)

        hidden_states = self.attentionBlock(hidden_states)

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