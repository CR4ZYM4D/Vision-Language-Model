import torch
import torch.nn as nn
import torch.nn.functional as f 

#moving config class here

class VisionModelConfig:
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

    def __init__(self, attention_heads, vector_dimension, linear_dimension, image_size, num_channels, patch_size, num_layers, dropout, normalization_constant, num_image_tokens):
        
        self.attention_heads = attention_heads
        self.vector_dimension = vector_dimension
        self.linear_dimension = linear_dimension
        self.image_size = image_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalization_constant = normalization_constant
        self.num_image_tokens = num_image_tokens

# class for the embedding layer of the vision transformer model

class VisionEmbedding(nn.Module):

    def __init__(self, config: VisionModelConfig):
        super().__init__()

        self.config = config

        self.convolutor = nn.Conv2d(
            in_channels = self.config.num_channels,  # 3 channels per image R, G, B
            out_channels= self.config.vector_dimension, 
            kernel_size= self.config.patch_size,
            stride = self.config.patch_size
            )
        
        self.num_patches = (self.config.image_size // self.config.patch_size) ** 2

        self.positionalEmbedding = nn.Embedding(self.num_patches, self.config.vector_dimension)

        self.register_buffer("positions", torch.arange(self.num_patches).expand((1, -1)), persistent= False)

    def forward(self, hidden_state):

        # image tensors are of shape [ batch_size x num_channels x height x width ] 

        # converting them to tensors of shape [ batch_size x vector_dimension x (height / patch_size) x (width / patch_size)]
        # using the CNN

        image_embeddings = self.convolutor(hidden_state) # [ batch_size x vector_dimension x (height / patch_size) x (width / patch_size) ]

        image_embeddings = image_embeddings.flatten(2) # [ batch_size x vector_dimension x num_patches ]

        image_embeddings = image_embeddings.transpose(1, 2) # [ batch_size x num_patches x vector_dimension ]

        return image_embeddings + self.positionalEmbedding(self.positions)


# class for the Vision Encoder block MLP layer

class VisionMLP(nn.Module):

    def __init__(self, config: VisionModelConfig):
        super().__init__()

        self.config = config
        self.layer1 = nn.Linear(self.config.vector_dimension, self.config.linear_dimension)
        self.layer2 = nn.Linear(self.config.linear_dimension, self.config.vector_dimension)

    def forward(self, hidden_state):

        hidden_state = self.layer1(hidden_state)

        hidden_state = f.gelu(hidden_state, approximate= "tanh")

        return self.layer2(hidden_state)
    
# class for the attention Block. Since it is an image transformer and not a langugae one, there is no need of a causal mask

class VisionAttention(nn.Module):

    def __init__(self, config: VisionModelConfig):
        super().__init__()

        self.config = config
        self.num_heads = self.config.attention_heads
        self.dropout = nn.Dropout(self.config.dropout)
        self.vector_dimension = self.config.vector_dimension
        self.head_dimension = self.config.vector_dimension // self.config.attention_heads

        self.wQ = nn.Linear(self.vector_dimension, self.vector_dimension) # Query projection layer
        self.wK = nn.Linear(self.vector_dimension, self.vector_dimension) # Key projection layer
        self.wV = nn.Linear(self.vector_dimension, self.vector_dimension) # Value projection layer
        self.wO = nn.Linear(self.vector_dimension, self.vector_dimension) # Output projection layer

        self.scale = self.vector_dimension ** -0.5

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

    def __init__(self, config: VisionModelConfig):
        super().__init__()
        self.config = config
        self.layerNorm1 = nn.LayerNorm(self.config.vector_dimension, self.config.normalization_constant)
        self.layerNorm2 = nn.LayerNorm(self.config.vector_dimension, self.config.normalization_constant) 

        self.attentionBlock = VisionAttention(self.config.vector_dimension, self.config.attention_heads, self.config.dropout)

        self.MLP = VisionMLP(self.config.vector_dimension, self.config.linear_dimension)

    def forward(self, hidden_state):

        skip_connector = torch.clone(hidden_state)

        hidden_state = self.layerNorm1(hidden_state)

        hidden_state = self.attentionBlock(hidden_state)

        hidden_state = hidden_state + skip_connector

        skip_connector = hidden_state

        hidden_state = self.layerNorm2(hidden_state)

        return hidden_state + skip_connector

# class for the vision transformer encoder

class VisionEncoder(nn.Module):

    def __init__(self, config: VisionModelConfig):
        super().__init__()
        self.config = config
        self.vector_dimension = self.config.vector_dimension
        self.normalization_constant = self.config.normalization_constant
        self.attention_heads = self.config.attention_heads
        self.dropout= self.config.dropout
        self.linear_dimension = self.config.linear_dimension
        self.num_layers = self.config.num_layers

        self.encoderLayers = nn.ModuleList( [VisionEncoderBlock(self.vector_dimension, 
                                                                self.normalization_constant, 
                                                                self.attention_heads, 
                                                                self.dropout, 
                                                                self.linear_dimension)] for _ in self.num_layers )
        
    def forward(self, hidden_state):

        hidden_states = hidden_state

        for EncoderBlock in self.encoderLayers:

            hidden_states = EncoderBlock(hidden_states)

        return hidden_states


# class to make the vision Transformer Model

class VisionTransformer(nn.Module):

    def __init__(self, config: VisionModelConfig):
        super().__init__()
        self.config = config
        self.embeddingLayer = VisionEmbedding(self.config.vector_dimension, self.config.patch_size, self.config.num_channels, self.config.image_size)
        self.encoderLayer = VisionEncoder(self.config.vector_dimension, self.config.normalization_constant, self.config.attention_heads, self.config.dropout, self.config.linear_dimension, self.config.num_layers)
        self.postNormalizationLayer = nn.LayerNorm(self.config.vector_dimension, eps = self.config.normalization_constant)

    def forward(self, hidden_state):

        hidden_state = self.embeddingLayer(hidden_state)

        hidden_state = self.encoderLayer(hidden_state)

        hidden_state = self.postNormalizationLayer(hidden_state)

        return hidden_state