import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from VisionModel import VisionModelConfig, VisionTransformer

# class for model architecture to give result from prompt and/or image

class LanguageModelConfig():
     
    def __init__(self, 
                  vocab_size, 
                  vector_dimension, 
                  feedforward_dimension, 
                  num_layers, 
                  num_query_heads, 
                  num_key_value_heads, 
                  head_dimension = 256, 
                  max_position_embeddings = 8192, 
                  rms_normalization_constant = 1e-6, 
                  rope_theta = 10000, 
                  attention_bias = False, 
                  dropout = 0.0, 
                  pad_token_id = None, 
                  **kwargs):
        super().__init__()

        self.vocab_size = vocab_size 
        self.vector_dimension = vector_dimension 
        self.feedforward_dimension = feedforward_dimension
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads 
        self.head_dimension = head_dimension 
        self.max_position_embeddings = max_position_embeddings 
        self.rms_normalization_constant = rms_normalization_constant
        self.rope_theta = rope_theta 
        self.attention_bias = attention_bias 
        self.dropout = dropout
        self.pad_token_id = pad_token_id



class VisionLanguageModelConfig():
     
     def __init__(self, 
                  vision_model: VisionModelConfig = None, 
                  language_model: LanguageModelConfig = None,
                  ignore_index = -100,
                  image_token_index = 256000,
                  vocab_size = 257152,
                  image_projection_dimension = 2048, 
                  vector_dimension = 2048, 
                  pad_token_id = None, 
                  **kwargs):
          super().__init__()

          self.ignore_index = ignore_index
          self.image_token_index = image_token_index
          self.vocab_size = vocab_size
          self.image_projection_dimension = image_projection_dimension
          self.vector_dimension = vector_dimension
          self.pad_token_id = pad_token_id

          self.vision_model = VisionModelConfig(**vision_model)
          self.language_model = LanguageModelConfig(**language_model, pad_token_id = pad_token_id)

          self.vocab_size = self.language_model.vocab_size

          self.language_model.num_image_tokens = (self.vision_model.image_size // self.vision_model.patch_size) ** 2

          self.vision_model.projection_dimension = image_projection_dimension

class VisionLanguageModel(nn.Module):

    def __init__(self, config: VisionLanguageModelConfig):

        self.config = config

        self.vision_model = VisionTransformer(self.config.vision_model)

        self.projection_layer = MultiModalProjectionLayer(self.config) # linear layer that passes image embeddings so that their embedding dimension matches token embedding dimension
        
        self.vocab_size = self.config.vocab_size

        language_model = ConditionalDecoder(config.language_model)

        self.language_model = language_model

        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1

    def forward(self, input_ids: torch.LongTensor, pixel_values: torch.FloatTensor, attention_mask, kvCache):

            # making sure padding is right
            assert torch.all(attention_mask == 1), "The input cannot be padded"

            # getting embeddings of image prompt
            input_embeddings = self.language_model.getInputEmbeddings()(input_ids) 

            # getting image features from Vision Model
            image_features = self.vision_model(pixel_values.to(input_embeddings.dtype))

            # matching image embeddings dimension as that of embedding dimension
            image_features = self.projection_layer(image_features)

            # replacing embeddings of <image> token from input embeddings with actual image embeddings
            modified_embeddings, attention_mask, position_ids = mergeTokensWithImageEmbeddings(self, image_features, input_embeddings, input_ids, attention_mask, kvCache)

            outputs = self.language_model(attention_mask = attention_mask,
                                          input_embeddings = modified_embeddings,
                                          kvCache = kvCache,
                                          position_ids = position_ids)

            return outputs 