# %%
# from ast import main
# from dataclasses import dataclass
# from huggingface_hub import hf_hub_download
import torch
from nnsight import NNsight
# import json
from typing import Any, Optional, List, Union, Callable
from datasets import load_dataset
# from einops import rearrange
# from jaxtyping import Int, Float, jaxtyped
# from torch import Tensor
# import os
# from tqdm import tqdm
from transformers import GPT2LMHeadModel
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils
from enum import Enum
# import pandas as pd

from . import othello_engine_utils as othello_engine_utils


# %%
def to_device(data, device):
    """
    Recursively move tensors in a nested dictionary to CPU.
    """
    if isinstance(data, dict):
        # If it's a dictionary, apply recursively to each value
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        # If it's a list, apply recursively to each element
        return [to_device(item, device) for item in data]
    elif isinstance(data, torch.Tensor):
        # If it's a tensor, move it to CPU
        return data.to(device)
    else:
        # If it's neither, return it as is
        return data
    
# %%
def get_model(model_name: str, device: torch.device) -> NNsight:
    if model_name == "Baidicoot/Othello-GPT-Transformer-Lens":
        tf_model = HookedTransformer.from_pretrained("Baidicoot/Othello-GPT-Transformer-Lens")
        model = NNsight(tf_model).to(device)
        return model
    
    if model_name == "mntss/Othello-GPT":
        tf_model = HookedTransformer.from_pretrained_no_processing("Baidicoot/Othello-GPT-Transformer-Lens")
        state_dict = tl_utils.download_file_from_hf("mntss/Othello-GPT", "tl_model.pth")
        tf_model.load_state_dict(state_dict)
        model = NNsight(tf_model).to(device)
        return model

    if (
        model_name == "adamkarvonen/RandomWeights8LayerOthelloGPT2"
        or model_name == "adamkarvonen/RandomWeights8LayerChessGPT2"
    ):
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        model = NNsight(model).to(device)
        return model

    if model_name == "adamkarvonen/8LayerChessGPT2":
        # Old method of loading model from nanogpt weights
        # model = convert_nanogpt_model(
        #     f"{model_path}lichess_8layers_ckpt_no_optimizer.pt", torch.device(device)
        # )
        # tokenizer = NanogptTokenizer(meta_path=f"{model_path}meta.pkl")
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        model = NNsight(model).to(device)
        return model

    raise ValueError("Model not found.")

# %%
def construct_othello_dataset(
    custom_functions: list[Callable],
    n_inputs: int,
    split: str,
    max_str_length: int = 59,
    device: str = "cpu",
    precompute_dataset: bool = True,
) -> dict:
    dataset = load_dataset("adamkarvonen/othello_45MB_games", streaming=False)
    encoded_othello_inputs_bL = []
    decoded_othello_inputs_bL = []
    for i, example in enumerate(dataset[split]):
        if i >= n_inputs:
            break
        encoded_input = example["tokens"][:max_str_length]
        decoded_input = othello_engine_utils.to_string(encoded_input)
        encoded_othello_inputs_bL.append(encoded_input)
        decoded_othello_inputs_bL.append(decoded_input)

    data = {}
    data["encoded_inputs"] = encoded_othello_inputs_bL
    data["decoded_inputs"] = decoded_othello_inputs_bL

    if not precompute_dataset:
        return data

    for custom_function in custom_functions:
        print(f"Precomputing {custom_function.__name__}...")
        func_name = custom_function.__name__
        data[func_name] = custom_function(decoded_othello_inputs_bL)

    return to_device(data, device) # changed by zihangw
