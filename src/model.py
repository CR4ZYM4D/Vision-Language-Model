import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from VisionModel import VisionModelConfig, VisionTransformer

# class for model architecture to give result from prompt and/or image

class Model(nn.Module):

    def __init__(self, config: VisionLanguageModelConfig):

        self.config = config

        self.vision_model = VisionTransformer(self.config.vision_model)

        self.projection_layer = MultiModalProjectionLayer(self.config) # linear layer that passes image embeddings so that their embedding dimension matches token embedding dimension
        
        self.vocab_size = self.config.vocab_size

        language_model = ConditionalDecoder(config.text_config)

        self.language_model = language_model

        self.pad_token_id = config.pad_token_id if config.pad_toke_id is not None else -1

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