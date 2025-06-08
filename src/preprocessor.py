import torch
import torch.nn as nn
import numpy as np 
from PIL import Image
from typing import List, Tuple

# class and methods to preprocess the image if it is not of model comaptible size.
# Use tokenizer to tokenize prompt and send it all to model class

def resizeAndRescale(image: Image, size: Tuple[int, int], rescale_factor, mean, std, resampling_technique = None)-> np.ndarray:

    height,width = size
    resized_image = [image.resize((width,height), resample = resampling_technique)]

    resized_image = [np.array(resized_image)]

    rescaled_image = [(resized_image * rescale_factor).astype(np.float32)]

    mean = np.array(mean, dtype= np.float32)
    std = np.array(std, dtype = np.float32)

    normalized_image = [(rescaled_image - mean)/std]

    image_token = [normalized_image.transpose(2, 0, 1)]

    return image_token

def addImageTokenToPrompt(prompt: str, bos_token, image_token, sequence_length):

    return f"{image_token * sequence_length}{bos_token}{prompt}\n" # the paper mentions using \n as a seperator token and to make sure to tokenize it as a token of its own 

#mean and standard deviation for normalizing the images to a gaussian distribution vector
NORMALIZING_MEAN = [0.5,0.5,0.5]
NORMALIZING_STD = [0.5,0.5,0.5]

class PreProcessor():

    def __init__(self, tokenizer, image_size, num_image_tokens = None):

        self.image_size = image_size 
        self.num_image_tokens = num_image_tokens

        self.image_token = "<image>"

        self.rescale_factor = 1/255.0
        self.resampling = Image.Resampling.BICUBIC


        token_to_add = [self.image_token] # we only add image marking token but we can also add location and segmentation
                                          #token for object detection and segmentation
        
        tokenizer.add_special_tokens({"additional_special_tokens": token_to_add})

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)

        self.tokenizer = tokenizer

    def process(self, 
                prompts: List[str],
                images: List[Image.Image], 
                padding: str = "longest", 
                truncation: bool = True ):

        # type checking is important here because if we don't we may loop through each character of a single string
        assert len(prompts) ==1 & len(images) == 1, f"max 1 image and prompt allowed at once received {len(images)} images and {len(prompts)} prompts"

        images = [resizeAndRescale(image = image, 
                         size = (self.image_size,self.image_size), 
                         resampling_technique = self.resampling, rescale_factor = self.rescale_factor, 
                         mean = NORMALIZING_MEAN,
                         std = NORMALIZING_STD) for image in images] # returns tensor of images resized and rescaled 

        processed_images = np.stack(images, axis = 0) # stack images tensors one after the other

        processed_images = torch.tensor(processed_images) # turning numpy array to torch tensor for use

        prompt_strings = [ addImageTokenToPrompt(prompt = prompt, 
                                                    bos_token = self.tokenizer.bos_token, 
                                                    image_token = self.image_token, 
                                                    sequence_length = self.num_image_tokens)
                                                     for prompt in prompts ]
        
        tokens = self.tokenizer(
                    prompt_strings,
                    return_tensors = "pt", #return pytorch tensors
                    padding = padding,
                    truncaton = truncation
                )

        return {"hidden_state": processed_images, **tokens}