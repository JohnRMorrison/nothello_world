# %%
import einops
from jaxtyping import Bool, Float, Int
import torch as t
from transformer_lens import HookedTransformer
# from analysis.helper_fns import (
#     calculate_neuron_input_weights,
# )

# %%
def get_w_in(
    model: HookedTransformer,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> Float[t.Tensor, "d_model"]:
    """
    Returns the input weights for the given neuron.

    If normalize is True, the weight is normalized to unit norm.
    """
    w_in = model.W_in[layer, :, neuron].detach().clone()
    if normalize:
        w_in /= w_in.norm(dim=0, keepdim=True)
    return w_in

def get_w_out(
    model: HookedTransformer,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> Float[t.Tensor, "d_model"]:
    """
    Returns the output weights for the given neuron.

    If normalize is True, the weight is normalized to unit norm.
    """
    w_out = model.W_out[layer, neuron, :].detach().clone()
    if normalize:
        w_out /= w_out.norm(dim=0, keepdim=True)
    return w_out

def get_w_out_all(
    model: HookedTransformer,
    normalize: bool = False,
) -> Float[t.Tensor, "layer neuron d_model"]:
    """
    Returns the output weights for the given neuron.

    If normalize is True, the weight is normalized to unit norm.
    """
    w_out = model.W_out.detach().clone()
    if normalize:
        w_out /= w_out.norm(dim=-1, keepdim=True)
    return w_out

def get_w_U(
    model: HookedTransformer,
    normalize: bool = False,
) -> Float[t.Tensor, "d_model token_id"]:
    """
    Returns the W_U weights for the model.

    If normalize is True, the weight is normalized to unit norm.
    """
    w_U = model.W_U[:, 1:].detach().clone()  # Exclude the "pass" move
    if normalize:
        w_U /= w_U.norm(dim=0, keepdim=True)
    return w_U

def calculate_neuron_input_weights(
    model: HookedTransformer, probe: Float[t.Tensor, "d_model row col"], layer: int, neuron: int
) -> Float[t.Tensor, "rows cols"]:
    """
    Returns t.Tensor of the input weights for the given neuron, at each square on the board, projected
    along the corresponding probe directions.

    Assume probe directions are normalized. You should also normalize the model weights.
    """
    w_in = get_w_in(model, layer, neuron, normalize=True)

    return einops.einsum(w_in, probe, "d_model, d_model row col -> row col")

def calculate_neuron_output_weights(
    model: HookedTransformer, probe: Float[t.Tensor, "d_model row col"], layer: int, neuron: int
) -> Float[t.Tensor, "rows cols"]:
    """
    Returns t.Tensor of the output weights for the given neuron, at each square on the board,
    projected along the corresponding probe directions.

    Assume probe directions are normalized. You should also normalize the model weights.
    """
    w_out = get_w_out(model, layer, neuron, normalize=True)

    return einops.einsum(w_out, probe, "d_model, d_model row col -> row col")

# %%
def load_probes_and_normalize(n_layers, device):
    probe_dict = {i : t.load(
        f"probes/linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{i}.pth", map_location=str(device), weights_only="True"
    )['linear_probe'].squeeze() for i in range(n_layers)}

    probe_t = t.stack([probe_dict[i] for i in range(n_layers)], dim=0)  # [layer, d_model, row, col, options]
    mine_probe = probe_t[..., 0]
    empty_probe = probe_t[..., 1]
    theirs_probe = probe_t[..., 2]  # [layer, d_model, row, col]

    mine_probe_normalized = mine_probe / mine_probe.norm(dim=1, keepdim=True)
    empty_probe_normalized = empty_probe / empty_probe.norm(dim=1, keepdim=True)
    theirs_probe_normalized = theirs_probe / theirs_probe.norm(dim=1, keepdim=True)
    empty_probe_normalized[..., [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

    flipped_probe_dict = {i : t.load(
        f"probes/linear_probes_flipped/resid_{i}_flipped.pth", map_location=str(device), weights_only="True"
    ).squeeze() for i in range(n_layers)}

    flipped_probe_t = t.stack([flipped_probe_dict[i] for i in range(n_layers)], dim=0)  # [layer, d_model, row, col, options]

    flipped_probe = flipped_probe_t[..., 0]
    flipped_probe_normalized = flipped_probe / flipped_probe.norm(dim=1, keepdim=True)
    
    just_played_probe_dict = {i : t.load(
        f"probes/linear_probes_just_played/resid_{i}_played.pth", map_location=str(device), weights_only="True"
    ).squeeze() for i in range(n_layers)
    }

    just_played_probe = t.stack([just_played_probe_dict[i] for i in range(n_layers)], dim=0)  # [layer, d_model, row, col]
    just_played_probe_normalized = just_played_probe / just_played_probe.norm(dim=1, keepdim=True)
    just_played_probe_normalized[..., [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

    return {
        "mine": mine_probe_normalized,
        "empty": empty_probe_normalized,
        "theirs": theirs_probe_normalized,
        "flipped": flipped_probe_normalized,
        "just_played": just_played_probe_normalized,
    }
    # return (
    #     mine_probe_normalized, empty_probe_normalized, theirs_probe_normalized,
    #     flipped_probe_normalized, just_played_probe_normalized
    # )

# %%
def load_fold_probes_and_normalize(n_layers, device):
    probe_dict = {i : t.load(
        f"probes/linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{i}.pth", map_location=str(device), weights_only="True"
    )['linear_probe'].squeeze() for i in range(n_layers)}

    probe_t = t.stack([probe_dict[i] for i in range(n_layers)], dim=0)  # [layer, d_model, row, col, options]
    blank_probe = probe_t[..., 1] - (probe_t[..., 0] + probe_t[..., 2]) * 0.5  # [layer, d_model, row, col]
    my_probe = probe_t[..., 0] - probe_t[..., 2]  # [layer, d_model, row, col]

    blank_probe_normalized = blank_probe / blank_probe.norm(dim=1, keepdim=True)
    my_probe_normalized = my_probe / my_probe.norm(dim=1, keepdim=True)
    blank_probe_normalized[..., [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

    flipped_probe_dict = {i : t.load(
        f"probes/linear_probes_flipped/resid_{i}_flipped.pth", map_location=str(device), weights_only="True"
    ).squeeze() for i in range(n_layers)}

    flipped_probe_t = t.stack([flipped_probe_dict[i] for i in range(n_layers)], dim=0)  # [layer, d_model, row, col, options]

    flipped_probe = flipped_probe_t[..., 0] - flipped_probe_t[..., 1]  # [layer, d_model, row, col]
    flipped_probe_normalized = flipped_probe / flipped_probe.norm(dim=1, keepdim=True)

    just_played_probe_dict = {i : t.load(
        f"probes/linear_probes_just_played/resid_{i}_played.pth", map_location=str(device), weights_only="True"
    ).squeeze() for i in range(n_layers)
    }

    just_played_probe = t.stack([just_played_probe_dict[i] for i in range(n_layers)], dim=0)  # [layer, d_model, row, col]
    just_played_probe_normalized = just_played_probe / just_played_probe.norm(dim=1, keepdim=True)
    just_played_probe_normalized[..., [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

    return {
        "blank": blank_probe_normalized,
        "mine": my_probe_normalized,
        "flipped": flipped_probe_normalized,
        "just_played": just_played_probe_normalized,
    }
    # return (
    #     blank_probe_normalized, my_probe_normalized,
    #     flipped_probe_normalized, just_played_probe_normalized
    # )

# %%
def calculate_w_in_cossim_with_probes(
    model,
    probes: list[t.Tensor],
    layer: int,
    neuron: int,
    layer_offset: int = 0,
):
    matric_list = [
        calculate_neuron_input_weights(model, probe[layer - layer_offset], layer, neuron) for probe in probes.values()
    ]
    # matric_list = [
    #     calculate_neuron_input_weights(model, probe[layer - layer_offset], layer, neuron) for probe in probes
    # ]
    matrices = t.stack(matric_list, dim=0).detach().cpu().numpy()  # [n_probes, 8, 8]

    return matrices

# %%

