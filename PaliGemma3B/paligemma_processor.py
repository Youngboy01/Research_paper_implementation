# PaliGemma processor creates text-tokens with placeholders for vision tokens
from typing import Optional, Union, List, Dict, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    image = image.astype(np.float32)
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image


def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.dtype(np.float32)
) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def resize(
    image: Image.Image,
    size: Tuple[int, int],
    resample: Optional[Image.Resampling] = None,
    reducing_gap: Optional[int] = None,
) -> Image.Image:
    height, width = size
    resized_img = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_img


def process_image(
    images: List[Image.Image],
    size: Optional[Tuple[int, int]] = None,
    resample: Optional[Image.Resampling] = None,
    rescale_factor: Optional[float] = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    if size is None:
        raise ValueError("size must be provided as a tuple (height, width)")
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    images = [
        np.array(image) for image in images
    ]  # converts each image to numpoy array
    if rescale_factor is not None:
        images = [
            rescale(image, scale=rescale_factor) for image in images
        ]  # rescales pixel values between [0,1]
    if image_mean is not None and image_std is not None:
        images = [
            normalize(image, mean=image_mean, std=image_std) for image in images
        ]  # normalizing images to have mean 0 and std 1
    # Move the channel dimension to the first dimension. The model expects images in the format [Channel, Height, Width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_length, image_token):
    """
    text is tokenized normally, bos at the beggining of te sentence and new line token \n is appended
    We saw that text tokens are also prefixed by fixed no of <image> tokens.
    \n should be tokenized separately acc to paper but not done in huggingface implementation."""
    return f"{image_token * image_seq_length}{bos_token}{prefix_prompt}\n"


class ProcessorPaliGemma:
    IMAGE_TOKEN = "<image>"
    """PaliGemma also supports image segmentation and object detection 
        but right now we are not concerned with these two but since it exist 
        in the original codebase so just included it"""
    EXTRA_TOKENS = [f"<loc{i:0>4}>" for i in range(1024)] + [
        f"<seg{i:0>3}>" for i in range(128)
    ]

    def __init__(self, tokenizer, num_img_tokens: int, img_size: int):
        super().__init__()
        self.image_seq_len = num_img_tokens
        self.image_size = img_size
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        tokenizer.add_tokens(self.EXTRA_TOKENS)
        self.img_token_ids = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        self.add_bos_token = False
        self.add_eos_token = False
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        assert len(images) == 1 and len(text) == 1, (
            f"received {len(images)} images for {len(text)} prompts"
        )
        pixel_values = process_image(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )  # it is returnig a list of numpy arrays
        # now lets converts the multiple numpy arrays into a single one of shape [batch_size,channel,height,width]
        pixel_values = np.stack(pixel_values, axis=0)
        # now convert numpy array into tensor
        pixel_values = torch.tensor(pixel_values)

        # we know that we need to add image tokens to the prompt, so prepend a self.image_seq_length number of image tokens to the prompt
        input_str = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_length=self.image_seq_len,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]
        inputs = self.tokenizer(
            input_str, return_tensors="pt", padding=padding, truncation=truncation
        )# returns input_ids and attention_mask as tensors
        # we are not using any padding for images so attention mask will be all 1s
        # tokenizer coverts text into list of numbers representing positions in vocabulary
        return_data = {"pixel_values": pixel_values, **inputs}
        return return_data
