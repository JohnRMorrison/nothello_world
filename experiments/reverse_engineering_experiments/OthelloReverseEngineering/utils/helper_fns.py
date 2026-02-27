import torch as t
import numpy as np
import einops
from collections import defaultdict

from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.utils import to_numpy
from torch import Tensor
# from IPython.display import HTML, display
from jaxtyping import Bool, Float, Int
# from typing import Optional, List, Callable
from dataclasses import dataclass
from datasets import load_dataset
# import matplotlib.pyplot as plt
# from sklearn.tree import plot_tree

from . import arena_utils as arena_utils
# from simulate_activations_with_dts import (
#     compute_kl_divergence,
#     compute_top_n_accuracy,
# )


MIDDLE_SQUARES = [27, 28, 35, 36]
ALL_SQUARES = [i for i in range(64) if i not in MIDDLE_SQUARES]
tracer_kwargs = {"validate": True, "scan": True}


# %%
def get_neuron_decision_tree(data: dict, layer: int, neuron_idx: int, function_name: str):
    """Extract the decision tree for a specific neuron."""
    if layer not in data:
        raise ValueError(f"Layer {layer} not found in data. Available layers: {list(data.keys())}")
    
    if function_name not in data[layer]:
        available_funcs = list(data[layer].keys())
        raise ValueError(f"Function {function_name} not found. Available: {available_funcs}")
    
    multi_output_model = data[layer][function_name]['decision_tree']['model']
    
    if neuron_idx >= len(multi_output_model.estimators_):
        raise ValueError(f"Neuron {neuron_idx} not found. Max neuron index: {len(multi_output_model.estimators_) - 1}")
    
    neuron_tree = multi_output_model.estimators_[neuron_idx]
    r2_scores = data[layer][function_name]['decision_tree']['r2']
    neuron_r2 = r2_scores[neuron_idx]
    
    return neuron_tree, neuron_r2

# %%
def get_neuron_binary_decision_tree(data: dict, layer: int, neuron_idx: int, function_name: str):
    """Extract the decision tree for a specific neuron."""
    if layer not in data:
        raise ValueError(f"Layer {layer} not found in data. Available layers: {list(data.keys())}")
    
    if function_name not in data[layer]:
        available_funcs = list(data[layer].keys())
        raise ValueError(f"Function {function_name} not found. Available: {available_funcs}")
    
    multi_output_model = data[layer][function_name]['binary_decision_tree']['model']
    
    if neuron_idx >= len(multi_output_model.estimators_):
        raise ValueError(f"Neuron {neuron_idx} not found. Max neuron index: {len(multi_output_model.estimators_) - 1}")
    
    neuron_tree = multi_output_model.estimators_[neuron_idx]
    f1_scores = data[layer][function_name]['binary_decision_tree']['f1']
    neuron_f1 = f1_scores[neuron_idx]
    
    return neuron_tree, neuron_f1

# %%
def compute_kl_divergence(logits_clean_BLV, logits_patch_BLV):
    # Apply softmax to get probability distributions
    log_probs_clean_BLV = t.nn.functional.log_softmax(logits_clean_BLV, dim=-1)
    log_probs_patch_BLV = t.nn.functional.log_softmax(logits_patch_BLV, dim=-1)

    # Compute KL divergence
    kl_div_BLV = t.nn.functional.kl_div(
        log_probs_patch_BLV, log_probs_clean_BLV.exp(), reduction="none", log_target=False
    )

    # Sum over the vocabulary dimension
    kl_div_BL = kl_div_BLV.sum(dim=-1)

    return kl_div_BL

def compute_top_n_accuracy(
    logits_BLV: t.Tensor, valid_moves_BLRRC: t.Tensor
) -> tuple[float, float, float]:
    valid_moves_BLC = einops.rearrange(valid_moves_BLRRC, "b l r1 r2 c -> b l (r1 r2 c)")
    n_BL = einops.reduce(valid_moves_BLC, "B L C -> B L", "sum")

    # Get the shape of the logits tensor
    B, L, V = logits_BLV.shape

    # Create a mask for the top n logits
    top_n_mask = t.zeros_like(logits_BLV, dtype=t.bool)

    for b in range(B):
        for l in range(L):
            n = n_BL[b, l].int()
            _, top_n_indices = t.topk(logits_BLV[b, l], k=n)
            top_n_mask[b, l, top_n_indices] = True

    top_n_mask = top_n_mask.int()
    stoi_top_n_mask = t.zeros(B, L, (V + 4), dtype=t.int32, device=top_n_mask.device)

    # This is so cursed. OthelloGPT has D vocab 61 (ignoring center squares, with pass at idx 0)
    stoi_top_n_mask[:, :, :28] = top_n_mask[:, :, :28]
    stoi_top_n_mask[:, :, 30:36] = top_n_mask[:, :, 28:34]
    stoi_top_n_mask[:, :, 38:] = top_n_mask[:, :, 34:]

    pass_BL1 = t.zeros(B, L, 1, dtype=t.int32, device=valid_moves_BLC.device)

    valid_moves_with_pass_BLC = t.cat([pass_BL1, valid_moves_BLC], dim=-1)

    correct_BLC = valid_moves_with_pass_BLC * stoi_top_n_mask

    correct = correct_BLC.sum()
    total = valid_moves_with_pass_BLC.sum()
    accuracy = correct / total

    return correct.item(), total.item(), accuracy.item()

# %%
def dt_simulation_neuron_activation(
    data: dict, decision_trees: dict, n_layer: int, n_neuron: int, func_name: str, device
) -> dict[int, t.Tensor]:
    board_state_BLC = data[func_name]
    B, L, C = board_state_BLC.shape
    X = einops.rearrange(board_state_BLC, "b l c -> (b l) c").cpu().numpy()

    simulated_acts = dict()
    for layer in range(n_layer):
        sim_layer = []
        for neuron in range(n_neuron):
            decision_tree = decision_trees[layer][neuron]
            simulated_activations_BF = decision_tree.predict(X)
            simulated_activations_BF = t.tensor(
                simulated_activations_BF, dtype=t.float32
            )
            simulated_activations_BLF = einops.rearrange(
                simulated_activations_BF, "(b l) -> b l", b=B, l=L
            )
            sim_layer.append(simulated_activations_BLF)
        simulated_acts[layer] = t.stack(sim_layer, dim=-1).to(device)  # B L neurons
    return simulated_acts

# %%
def neuron_intervention(
    model,
    layers_neurons: dict[list],
    game_batch_BL: t.Tensor,
    ablation_method: str = "zero",
    simulated_acts: dict = None,
):
    allowed_methods = ["mean", "max", "zero", "dt"]
    # allowed_methods = ["zero"]
    assert ablation_method in allowed_methods, (
        f"Invalid ablation method. Must be one of {allowed_methods}"
    )

    mean_activations = {}
    max_activations = {}

    # Get clean logits and mean submodule activations
    with t.no_grad(), model.trace(game_batch_BL, **tracer_kwargs):
        for layer in range(model.cfg.n_layers):
            original_input_BLD = model.blocks[layer].mlp.hook_post.output
            if ablation_method == "mean":
                mean_activations[layer] = original_input_BLD.mean(dim=(0, 1)).save()
            elif ablation_method == "max":
                # max_activations_temp = original_input_BLD.max(dim=0).values
                max_activations[layer] = original_input_BLD.max(dim=(0, 1)).values.save()
            else:
                # No need to do anything for other ablations, just save the original input
                pass
        logits_clean_BLV = model.unembed.output.save()
    
    with t.no_grad(), model.trace(game_batch_BL, **tracer_kwargs):
        for layer, neuron_indices in layers_neurons.items():
            original_input_BLD = model.blocks[layer].mlp.hook_post.output
            if ablation_method == "mean":
                original_input_BLD[:, :, neuron_indices] = mean_activations[layer][neuron_indices]
            elif ablation_method == "max":
                original_input_BLD[:, :, neuron_indices] = max_activations[layer][neuron_indices]
            elif ablation_method == "zero":
                original_input_BLD[:, :, neuron_indices] = 0.0
            elif ablation_method == "dt":
                original_input_BLD[:, :, neuron_indices] = simulated_acts[layer][:, :, neuron_indices]
        
        logits_patch_BLV = model.unembed.output.save()
    
    return logits_clean_BLV, logits_patch_BLV

# %%
def calculate_ablation_scores_game_move(model, layers_neurons, focus_games_id, focus_legal_moves, move, ablation_method = "zero"):
    logits_clean_BLV, logits_patch_BLV = neuron_intervention(
        model,
        layers_neurons=layers_neurons,
        game_batch_BL=focus_games_id,
        ablation_method=ablation_method,
    )

    logits_clean_BLV_move = logits_clean_BLV[:, move].unsqueeze(1) # [1, 1, ids]
    logits_patch_BLV_move = logits_patch_BLV[:, move].unsqueeze(1) # [1, 1, ids]
    focus_legal_moves_move = focus_legal_moves[:, move].unsqueeze(1).unsqueeze(-1)  # [1, 1, row, col, 1]

    kl_div_BL = compute_kl_divergence(logits_clean_BLV_move, logits_patch_BLV_move)

    _, _, clean_accuracy = compute_top_n_accuracy(
        logits_clean_BLV_move, focus_legal_moves_move
    )

    _, _, patch_accuracy = compute_top_n_accuracy(
        logits_patch_BLV_move, focus_legal_moves_move
    )

    return kl_div_BL.mean().item(), clean_accuracy, patch_accuracy

# %%
def calculate_ablation_scores_square(model, layers_neurons, board_seqs_id, valid_move_square_mask, valid_move_number, token_id, ablation_method = "zero"):
    logits_clean_BLV, logits_patch_BLV = neuron_intervention(
        model,
        layers_neurons=layers_neurons,
        game_batch_BL=board_seqs_id,
        ablation_method=ablation_method,
    )

    kl_div_BL = compute_kl_divergence(logits_clean_BLV, logits_patch_BLV)

    logits_clean_rank_token = (logits_clean_BLV > logits_clean_BLV[...,token_id].unsqueeze(-1)).sum(-1)
    clean_total = valid_move_square_mask.sum()
    clean_correct = ((logits_clean_rank_token < valid_move_number) * valid_move_square_mask).sum()
    clean_accuracy = clean_correct / clean_total

    logits_patch_rank_token = (logits_patch_BLV > logits_patch_BLV[...,token_id].unsqueeze(-1)).sum(-1)
    patch_total = valid_move_square_mask.sum()
    patch_correct = ((logits_patch_rank_token < valid_move_number) * valid_move_square_mask).sum()
    patch_accuracy = patch_correct / patch_total

    # logits_patch_order = t.argsort(logits_patch_BLV, dim=-1, descending=True)
    # logits_patch_rank = t.argsort(logits_patch_order, dim=-1)
    # logits_patch_rank_token = logits_patch_rank[..., token_id]  # [game, seq]

    return kl_div_BL.mean().item(), clean_accuracy.item(), patch_accuracy.item()

def calculate_ablation_scores_square_probability(model, layers_neurons, board_seqs_id, valid_move_square_mask, valid_move_number, token_id, ablation_method = "zero", threshold = 0.1, simulated_acts = None):
    logits_clean_BLV, logits_patch_BLV = neuron_intervention(
        model,
        layers_neurons=layers_neurons,
        game_batch_BL=board_seqs_id,
        ablation_method=ablation_method,
        simulated_acts=simulated_acts,
    )
    valid_move_square_mask_bool = valid_move_square_mask.to(dtype=bool)
    kl_div_BL = compute_kl_divergence(logits_clean_BLV, logits_patch_BLV)

    logits_clean_BLV_sm = logits_clean_BLV.softmax(dim=-1)[...,token_id]
    logits_patch_BLV_sm = logits_patch_BLV.softmax(dim=-1)[...,token_id]

    clean_flat = logits_clean_BLV_sm[valid_move_square_mask_bool]
    patch_flat = logits_patch_BLV_sm[valid_move_square_mask_bool]
    valid_move_number_flat = valid_move_number[valid_move_square_mask_bool]

    play_total = valid_move_square_mask_bool.sum()
    clean_accuracy = (clean_flat > 1 / valid_move_number_flat * threshold).sum() / play_total
    patch_accuracy = (patch_flat > 1 / valid_move_number_flat * threshold).sum() / play_total

    return kl_div_BL.mean().item(), clean_accuracy.item(), patch_accuracy.item()

@dataclass
class InterventionMetrics:
    kl_div: float
    logit_diff: float
    prob_diff: float
    clean_accuracy: float
    corrupted_accuracy: float
    accuracy_diff: float
    below_1_percent: float
    below_5_percent: float
    below_10_percent: float

def calculate_ablation_scores_square_all(model, layers_neurons, board_seqs_id, valid_move_square_mask, valid_move_number, token_id, ablation_method = "zero"):
    logits_clean_BLV, logits_patch_BLV = neuron_intervention(
        model,
        layers_neurons=layers_neurons,
        game_batch_BL=board_seqs_id,
        ablation_method=ablation_method,
    )
    
    valid_move_square_mask_bool = valid_move_square_mask.to(dtype=bool)
    kl_div_BL = compute_kl_divergence(logits_clean_BLV, logits_patch_BLV)

    logits_clean_rank_token = (logits_clean_BLV > logits_clean_BLV[...,token_id].unsqueeze(-1)).sum(-1)
    clean_total = valid_move_square_mask.sum()
    clean_correct = ((logits_clean_rank_token < valid_move_number) * valid_move_square_mask).sum()
    clean_accuracy_topk = clean_correct / clean_total

    logits_patch_rank_token = (logits_patch_BLV > logits_patch_BLV[...,token_id].unsqueeze(-1)).sum(-1)
    patch_total = valid_move_square_mask.sum()
    patch_correct = ((logits_patch_rank_token < valid_move_number) * valid_move_square_mask).sum()
    patch_accuracy_topk = patch_correct / patch_total

    ave_logit_diff = (logits_clean_BLV - logits_patch_BLV)[...,token_id][valid_move_square_mask_bool].mean()

    logits_clean_BLV_sm = logits_clean_BLV.softmax(dim=-1)[...,token_id]
    logits_patch_BLV_sm = logits_patch_BLV.softmax(dim=-1)[...,token_id]

    clean_flat = logits_clean_BLV_sm[valid_move_square_mask_bool]
    patch_flat = logits_patch_BLV_sm[valid_move_square_mask_bool]
    valid_move_number_flat = valid_move_number[valid_move_square_mask_bool]

    ave_prob_diff = (clean_flat - patch_flat).mean()

    play_total = valid_move_square_mask_bool.sum()
    # clean_accuracy = (clean_flat > 1 / valid_move_number_flat * threshold).sum() / play_total
    # patch_accuracy = (patch_flat > 1 / valid_move_number_flat * threshold).sum() / play_total

    below_1_percent_corrupted = (patch_flat < 1 / valid_move_number_flat * 0.01).sum() / play_total
    below_5_percent_corrupted = (patch_flat < 1 / valid_move_number_flat * 0.05).sum() / play_total
    below_10_percent_corrupted = (patch_flat < 1 / valid_move_number_flat * 0.1).sum() / play_total

    return InterventionMetrics(
        kl_div=kl_div_BL.mean().item(),
        logit_diff=ave_logit_diff.item(),
        prob_diff=ave_prob_diff.item(),
        clean_accuracy=clean_accuracy_topk.item(),
        corrupted_accuracy=patch_accuracy_topk.item(),
        accuracy_diff=(clean_accuracy_topk - patch_accuracy_topk).item(),
        below_1_percent=below_1_percent_corrupted.item(),
        below_5_percent=below_5_percent_corrupted.item(),
        below_10_percent=below_10_percent_corrupted.item(),
    )
    # metrics = {
    #     "kl_div": kl_div_BL.mean().item(),
    #     "clean_accuracy_topk": clean_accuracy_topk.item(),
    #     "patch_accuracy_topk": patch_accuracy_topk.item(),
    #     "ave_logit_diff": ave_logit_diff.item(),
    #     "ave_prob_diff": ave_prob_diff.item(),
    #     "below_1_percent_corrupted": below_1_percent_corrupted.item(),
    #     "below_5_percent_corrupted": below_5_percent_corrupted.item(),
    #     "below_10_percent_corrupted": below_10_percent_corrupted.item(),
    # }
    
    # return metrics

# %%
# def visualize_decision_tree(tree_model, neuron_idx: int, layer: int, r2_score: float,
#                           feature_names: List[str], max_depth: Optional[int] = None,
#                           save_path: Optional[str] = None):
#     """Visualize a decision tree for a specific neuron."""
#     plt.figure(figsize=(20, 12))
    
#     plot_tree(
#         tree_model,
#         feature_names=feature_names,
#         filled=True,
#         rounded=True,
#         fontsize=8,
#         max_depth=max_depth
#     )
    
#     plt.title(f"Decision Tree for Layer {layer}, Neuron {neuron_idx}\nR² Score: {r2_score:.4f}", 
#               fontsize=16, pad=20)
    
#     if save_path:
#         plt.savefig(f"{save_path}/dt_layer_{layer}_neuron_{neuron_idx}.png", dpi=300, bbox_inches='tight')
#         print(f"Saved visualization to {save_path}")
    
#     plt.show()

# %%
def get_board_states_and_legal_moves(
    games_square: Int[Tensor, "n_games n_moves"],
) -> tuple[
    Int[Tensor, "n_games n_moves rows cols"],
    Int[Tensor, "n_games n_moves rows cols"],
    list,
]:
    """
    Returns the following:
        states:                 (n_games, n_moves, 8, 8): tensor of board states after each move
        legal_moves:            (n_games, n_moves, 8, 8): tensor of 1s for legal moves, 0s for illegal moves
        legal_moves_annotation: (n_games, n_moves, 8, 8): list containing strings of "o" for legal moves (for plotting)
    """
    # Create tensors to store the board state & legal moves
    n_games, n_moves = games_square.shape
    states = t.zeros((n_games, n_moves, 8, 8), dtype=t.int32)
    legal_moves = t.zeros((n_games, n_moves, 8, 8), dtype=t.int32)

    # Loop over each game, populating state & legal moves tensors after each move
    for n in range(n_games):
        board = arena_utils.OthelloBoardState()
        for i in range(n_moves):
            board.umpire(games_square[n, i].item())
            states[n, i] = t.from_numpy(board.state)
            legal_moves[n, i].flatten()[board.get_valid_moves()] = 1

    # Convert legal moves to annotation
    legal_moves_annotation = np.where(to_numpy(legal_moves), "o", "")

    return states, legal_moves, legal_moves_annotation

# %%
# def plot_probe_outputs(
#     cache: ActivationCache,
#     probe_dict : dict[int, Tensor],
#     layer: int,
#     game_index: int,
#     move: int,
#     title: str = "Probe outputs",
# ):
#     linear_probe = probe_dict[layer]
#     residual_stream = cache["resid_post", layer][game_index, move]
#     probe_out = einops.einsum(
#         residual_stream,
#         linear_probe,
#         "d_model, d_model row col options -> options row col",
#     )

#     arena_utils.plot_board_values(
#         probe_out.softmax(dim=0),
#         title=title,
#         width=900,
#         height=400,
#         board_titles=["P(Mine)", "P(Empty)", "P(Their's)"],
#         # text=BOARD_LABELS_2D,
#     )


# %% PLOTTING LOG PROBS
# First 10 moves of game 0
# sample_input = t.tensor(test_data["encoded_inputs"][0][:10]).to(device)
# with model.trace(sample_input):
#     logits = model.unembed.output.save()
# logprobs = logits.log_softmax(dim=-1)

# logprobs_board = t.full(size=(8, 8), fill_value=-13.0, device=device)
# logprobs_board.flatten()[ALL_SQUARES] = logprobs[
#     0, 0, 1:
# ]  # the [1:] is to filter out logits for the "pass" move

# arena_utils.plot_board_values(logprobs_board, title="Example Log Probs", width=500)

# %% PLOTTING LOG PROBS with ANNOTATED TOKEN IDS and BOARD LABELS
# TOKEN_IDS_2D = np.array(
#     [str(i) if i in ALL_SQUARES else "" for i in range(64)]
# ).reshape(8, 8)
# BOARD_LABELS_2D = np.array(
#     ["ABCDEFGH"[i // 8] + f"{i % 8}" for i in range(64)]
# ).reshape(8, 8)

# print(TOKEN_IDS_2D)
# print(BOARD_LABELS_2D)

# arena_utils.plot_board_values(
#     t.stack([logprobs_board, logprobs_board]),  # shape (2, 8, 8)
#     title="Example Log Probs (with annotated token IDs)",
#     width=800,
#     text=np.stack([TOKEN_IDS_2D, BOARD_LABELS_2D]),  # shape (2, 8, 8)
#     board_titles=["Labelled by token ID", "Labelled by board label"],
# )

# %% PLOTTING LOG PROBS (10 MOVES)
# logprobs_multi_board = t.full(size=(10, 8, 8), fill_value=-13.0, device=device)
# logprobs_multi_board.flatten(1, -1)[:, ALL_SQUARES] = logprobs[
#     0, :, 1:
# ]  # we now do all 10 moves at once

# arena_utils.plot_board_values(
#     logprobs_multi_board,
#     title="Example Log Probs",
#     width=1000,
#     boards_per_row=5,
#     board_titles=[f"Logprobs after move {i}" for i in range(1, 11)],
# )

# %% PLOTTING BOARD STATES AND LEGAL MOVES (10 MOVES)
# board_states = t.zeros((10, 8, 8), dtype=t.int32)
# legal_moves = t.zeros((10, 8, 8), dtype=t.int32)

# board = arena_utils.OthelloBoardState()
# for i, token_id in enumerate(sample_input.squeeze()):
#     # board.umpire takes a square index (i.e. from 0 to 63) and makes a move on the board
#     board.umpire(arena_utils.id_to_square(token_id))

#     # board.state gives us the 8x8 numpy array of 0 (blank), -1 (black), 1 (white)
#     board_states[i] = t.from_numpy(board.state)

#     # board.get_valid_moves() gives us a list of the indices of squares that are legal to play next
#     legal_moves[i].flatten()[board.get_valid_moves()] = 1

# # Turn `legal_moves` into strings, with "o" where the move is legal and empty string where illegal
# legal_moves_annotation = np.where(to_numpy(legal_moves), "o", "").tolist()

# arena_utils.plot_board_values(
#     board_states,
#     title="Board states",
#     width=1000,
#     boards_per_row=5,
#     board_titles=[f"State after move {i}" for i in range(1, 11)],
#     text=legal_moves_annotation,
# )
