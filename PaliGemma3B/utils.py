from modelling_gemma import PaliGemmaConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os


def load_hf_model(
    model_path: str, device: str
) -> Tuple[PaliGemmaConditionalGeneration, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # find all safetensors files in the model_path
    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    # load them one by one in tensor dictionary
    tensors = {}
    for safetensor_file in safetensor_files:
        with safe_open(safetensor_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    # load config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)
    model = PaliGemmaConditionalGeneration(config).to(device)
    # load state dict
    model.load_state_dict(tensors, strict=False)
    # tie weights
    model.tie_weights()
    return (model, tokenizer)
