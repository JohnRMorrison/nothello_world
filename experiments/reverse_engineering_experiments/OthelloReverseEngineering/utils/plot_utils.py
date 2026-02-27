# %%
import os
import sys
from pathlib import Path
from typing import Callable, Optional
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import graphviz
from sklearn.tree import export_graphviz
# from sklearn.tree import plot_tree
import torch as t
from transformer_lens.utils import to_numpy

from . import arena_utils as arena_utils
from probe_utils import (
    calculate_w_in_cossim_with_probes,
)
from helper_fns import (
    get_board_states_and_legal_moves,
)

# %%
def plot_decision_tree(
    tree_model,
    feature_names: list[str],
    # function_name: str,
    layer: int,
    neuron: int,
    # i_k: int,
    # r2_score: float,
    save_path: Path = None,
    figure_type: str = "png",
):
    dot_data = export_graphviz(
        tree_model,
        out_file=None,
        feature_names=feature_names,
        filled=True, rounded=True,
        special_characters=True,
        proportion=False,   # scale node size by samples
        max_depth=5,
        impurity=False,   # removes "mse" or "variance"
    )
    dot_data = re.sub(r'samples = \d+', '', dot_data)
    graph = graphviz.Source(dot_data)
    # graph.graph_attr.update(label=f"Decision Tree (Rank {i_k}: L{layer}N{neuron})\nR² Score: {r2_score:.4f}", labelloc='top', fontsize='16')
    # graph.render("regression_tree")  # saves PDF/PNG
    # graph
    # os.makedirs(figure_save_path:=f"figures/decision_tree_0826_features_pdf_5/{function_name}", exist_ok=True)
    # graph.render(f"{figure_save_path}/dt_layer_rank_{i_k}_L{layer}N{neuron}", format="pdf", cleanup=True)

    os.makedirs(save_path, exist_ok=True)
    graph.render(save_path / f"dt_L{layer}N{neuron}", format="figure_type", cleanup=True)

    # fig, ax = plt.subplots(figsize=(20, 12))

    # plot_tree(
    #     tree_model,
    #     feature_names=feature_names,
    #     filled=True,
    #     rounded=True,
    #     fontsize=8,
    #     max_depth=max_depth
    # )
    
    # ax.set_title(f"Decision Tree (Rank {i_k}: L{layer}N{neuron})\nR² Score: {r2_score:.4f}", 
    #           fontsize=16, pad=20)
    # fig.savefig(f"figures/decision_tree/dt_layer_rank_{i_k}_L{layer}N{neuron}.png", dpi=300, bbox_inches='tight')

# %%
def plot_board_states(
    data: list,
    game_index: int,
    move: int,
    save_path: Path = None,
    figure_type: str = "png",
):
    board_seqs_square = t.tensor(data["decoded_inputs"])
    board_seqs_square = board_seqs_square[game_index, :move+1].unsqueeze(0) # [move, 8, 8]

    board_states, legal_moves, _ = get_board_states_and_legal_moves(board_seqs_square)

    sqaure_label = arena_utils.to_board_label(board_seqs_square[0, move])
    
    fig = arena_utils.plot_board_values(
        board_states[0, move],
        width=500,
        title=f"After move {move} ({sqaure_label}), {'white' if move % 2 == 0 else 'black'} to play",
        text=np.where(to_numpy(legal_moves[0, move]), "o", "").tolist(),
    )

    os.makedirs(save_path, exist_ok=True)
    fig.write_image(save_path / f"game{game_index}_move{move}.{figure_type}")

# %%
def plot_board_probes(
    model,
    probes: list[t.Tensor],
    titles: list[str],
    layer: int,
    neuron: int,
    save_path: Path = None,
    figure_type: str = "png",
):
    assert len(probes) == len(titles)

    titles = [f"{title} for L{layer}N{neuron}" for title in titles]

    matrices = calculate_win_cossim_with_probes(
        model, probes, layer, neuron, layer_offset=1,
    )

    fig = arena_utils.plot_board_values(
        matrices,
        title=f"Input weights cosine similarity with the probe for neuron L{layer}N{neuron}",
        board_titles=titles,
        boards_per_row=3,
        width=325*3,
        height=760,
    )

    os.makedirs(save_path, exist_ok=True)
    fig.write_image(save_path / f"L{layer}N{neuron}.{figure_type}")

    # mine_probe_normalized, empty_probe_normalized, theirs_probe_normalized, flipped_probe_normalized, just_played_probe_normalized = probes

    # titles = [
    #     f"Mine In L{layer}N{neuron}", f"Empty In L{layer}N{neuron}", f"Theirs In L{layer}N{neuron}",
    #     f"Flipped In L{layer}N{neuron}", f"Just Played In L{layer}N{neuron}",
    # ]
