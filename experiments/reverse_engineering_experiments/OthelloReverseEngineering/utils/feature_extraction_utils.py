# %%
from tqdm import trange
from typing import Optional, List
from collections import defaultdict

from sklearn.tree import _tree
from skimage.filters import threshold_otsu

# %%
def create_pbs_feature_names(n_features: int) -> List[str]:
    """Create feature names based on the actual feature structure:
    (192) + (64 + 64 + 5) + (64) = 389 dimensional vector
    - Board state: 192 one-hot (8x8x3 mine/empty/theirs)
    - Last move: 64 one-hot move + 64 pre-occupied + 5 coordinates  
    - Flipped moves: 64 binary encoding of flipped squares
    
    Square notation: A0-H7 where A0 is top-left, H7 is bottom-right
    """
    feature_names = []
    idx = 0
    
    # First 192: Board state (8x8x3 = mine/empty/theirs)
    for square_idx in range(min(64, (n_features - idx) // 3)):
        row = square_idx // 8  
        col = square_idx % 8
        square = chr(ord('A') + row) + str(col)
        
        # Add the 3 states for this square
        if idx < n_features:
            feature_names.append(f"prev_board_{square}_theirs")
            idx += 1
        if idx < n_features:
            feature_names.append(f"prev_board_{square}_empty") 
            idx += 1
        if idx < n_features:
            feature_names.append(f"prev_board_{square}_mine")
            idx += 1

    # Next 64: Last move one-hot encoding (A0-H7)
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + row) + str(col)
        feature_names.append(f"{square}_just_played")
        idx += 1
    
    # Next 64: Pre-occupied squares (A0-H7)  
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + row) + str(col)
        feature_names.append(f"{square}_pre_occupied")
        idx += 1
    
    # Next 5: Move coordinates and player info
    coord_names = ["move_row", "move_col", "move_number", "white_played", "black_played"]
    for i in range(min(5, n_features - idx)):
        feature_names.append(coord_names[i])
        idx += 1
    
    # Last 64: Flipped squares (A0-H7)
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + row) + str(col)
        feature_names.append(f"{square}_flipped")
        idx += 1
    
    # Add any remaining features as generic (shouldn't happen with 389 total)
    while idx < n_features:
        feature_names.append(f"Feature_{idx}")
        idx += 1
    
    return feature_names


def create_bs_feature_names(n_features: int) -> List[str]:
    """Create feature names based on the actual feature structure:
    (192) + (64 + 64 + 5) + (64) = 389 dimensional vector
    - Board state: 192 one-hot (8x8x3 mine/empty/theirs)
    - Last move: 64 one-hot move + 64 pre-occupied + 5 coordinates  
    - Flipped moves: 64 binary encoding of flipped squares
    
    Square notation: A0-H7 where A0 is top-left, H7 is bottom-right
    """
    feature_names = []
    idx = 0
    
    # First 192: Board state (8x8x3 = mine/empty/theirs)
    for square_idx in range(min(64, (n_features - idx) // 3)):
        row = square_idx // 8  
        col = square_idx % 8
        square = chr(ord('A') + row) + str(col)
        
        # Add the 3 states for this square
        if idx < n_features:
            feature_names.append(f"{square}_mine")
            idx += 1
        if idx < n_features:
            feature_names.append(f"{square}_empty") 
            idx += 1
        if idx < n_features:
            feature_names.append(f"{square}_theirs")
            idx += 1

    # Next 64: Last move one-hot encoding (A0-H7)
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + row) + str(col)
        feature_names.append(f"{square}_just_played")
        idx += 1
    
    # Next 64: Pre-occupied squares (A0-H7)  
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + row) + str(col)
        feature_names.append(f"{square}_pre_occupied")
        idx += 1
    
    # Next 5: Move coordinates and player info
    coord_names = ["move_row", "move_col", "move_number", "white_played", "black_played"]
    for i in range(min(5, n_features - idx)):
        feature_names.append(coord_names[i])
        idx += 1
    
    # Last 64: Flipped squares (A0-H7)
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + row) + str(col)
        feature_names.append(f"{square}_flipped")
        idx += 1
    
    # Add any remaining features as generic (shouldn't happen with 389 total)
    while idx < n_features:
        feature_names.append(f"Feature_{idx}")
        idx += 1
    
    return feature_names


def create_bs_flipped_played_feature_names(n_features: int) -> List[str]:
    feature_names = []
    idx = 0
    # First 192: Board state (8x8x3 = mine/empty/theirs)
    for square_idx in range(min(64, (n_features - idx) // 3)):
        row = square_idx // 8  
        col = square_idx % 8
        square = chr(ord('A') + row) + str(col)
        
        # Add the 3 states for this square
        if idx < n_features:
            feature_names.append(f"{square}_mine")
            idx += 1
        if idx < n_features:
            feature_names.append(f"{square}_empty") 
            idx += 1
        if idx < n_features:
            feature_names.append(f"{square}_theirs")
            idx += 1
    
    # Next 64: Flipped squares (A0-H7)
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + row) + str(col)
        feature_names.append(f"{square}_flipped")
        idx += 1

    # Next 64: Last move one-hot encoding (A0-H7)
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + row) + str(col)
        feature_names.append(f"{square}_just_played")
        idx += 1
    
    return feature_names


def create_bs_flipped_played_valid_move_feature_names(n_features: int) -> List[str]:
    feature_names = create_bs_flipped_played_feature_names(n_features - 64)
    idx = n_features - 64

    # Last 64: Valid move squares (A0-H7)
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + row) + str(col)
        feature_names.append(f"{square}_is_valid_move")
        idx += 1

    return feature_names


def get_feature_names_cont_dt():
    """
    Generate feature names for all projections.
    
    Returns list of feature names in order:
    - 64 projections for mine - theirs (all squares)
    - 60 projections for blank (excluding middle 4: D3, D4, E3, E4)
    - 64 projections for flipped (all squares)
    - 60 projections for placed (excluding middle 4)
    """
    feature_names = []
    
    # Helper to get square name from row/col
    def get_square_name(row, col):
        row_letter = chr(ord('A') + row)
        return f"{row_letter}{col}"
    
    # 1. Mine - Theirs (all 64 squares)
    for row in range(8):
        for col in range(8):
            square = get_square_name(row, col)
            feature_names.append(f"{square}_mine-theirs")
    
    # 2. Blank (60 squares, excluding D3, D4, E3, E4)
    middle_squares = {(3, 3), (3, 4), (4, 3), (4, 4)}  # D3, D4, E3, E4
    for row in range(8):
        for col in range(8):
            if (row, col) not in middle_squares:
                square = get_square_name(row, col)
                feature_names.append(f"{square}_blank")
    
    # 3. Flipped (all 64 squares)
    for row in range(8):
        for col in range(8):
            square = get_square_name(row, col)
            feature_names.append(f"{square}_flipped")
    
    # 4. Placed (60 squares, excluding middle 4)
    for row in range(8):
        for col in range(8):
            if (row, col) not in middle_squares:
                square = get_square_name(row, col)
                feature_names.append(f"{square}_just_played")
    
    return feature_names


def create_feature_names(n_features: int, function_name) -> List[str]:
    if function_name == "games_batch_to_input_tokens_flipped_pbs_classifier_input_BLC":
        return create_pbs_feature_names(n_features)
    elif function_name == "games_batch_to_input_tokens_flipped_bs_classifier_input_BLC":
        return create_bs_feature_names(n_features)
    elif function_name == "games_batch_to_board_state_flipped_played_BLC":
        return create_bs_flipped_played_feature_names(n_features)
    elif function_name == "games_batch_to_board_state_flipped_played_valid_move_BLC":
        return create_bs_flipped_played_valid_move_feature_names(n_features)
    else:
        raise ValueError(f"Unknown function name: {function_name}. Cannot create feature names.")
    

# %%
def extract_rules_features_from_binary_dt(
    num_layers,
    num_neurons, 
    binary_decision_trees,
    f1_scores,
    binary_feature_names,
    f1_threshold=0.7,
):
    dt_rules = defaultdict(dict)

    for layer in trange(num_layers):
        for neuron in range(num_neurons):
            if neuron not in binary_decision_trees[layer]:
                continue
            # binary_tree_model = binary_decision_trees[layer][custom_function_name]['binary_decision_tree']['model'].estimators_[neuron]
            # f1 = binary_decision_trees[layer][custom_function_name]['binary_decision_tree']['f1'][neuron].item()

            binary_tree_model = binary_decision_trees[layer][neuron]
            f1 = f1_scores[layer][neuron]
            # if (f1_threshold is not None) and f1 < f1_threshold:
            #     continue

            (rules, pred_strengths, samples_per_rule, _), _ = extract_and_rules_binary_dt(binary_tree_model, binary_feature_names, target_class=1, value_threshold=0.7)

            sorted_rules = sorted(
                zip(rules, pred_strengths, samples_per_rule),
                key=lambda x: (x[2], x[1]),  # sort by samples_per_rule, then pred_strength
                reverse=True
            )

            filter_min_samples = binary_tree_model.tree_.n_node_samples[0].item() / 59 * .05
            filtered_rules = [(rule_infer(rule), strength, samples) for rule, strength, samples in sorted_rules if samples >= filter_min_samples]

            rule_list = []
            filtered_features = set()
            filtered_direct_features = set()
            for rule, strength, samples in filtered_rules:
                # print(f"Rule: {rule}\n\t(Strength: {strength:.2f}, Samples: {samples})")
                direct_feat_infered = set(rule.split(" AND "))
                feature_inferred = {feat.split("(")[-1].split(")")[0].split(" ")[-1] for feat in direct_feat_infered}
                
                rule_list.append(rule)
                filtered_features.update(feature_inferred)
                filtered_direct_features.update(direct_feat_infered)
                
            dt_rules[layer][neuron] = {
                "dt_rules": rule_list,
                "dt_filtered_features": filtered_features,
                "dt_filtered_directional_features": filtered_direct_features,
                "dt_f1": f1,
            }

    return dt_rules

# %%
def extract_and_rules_binary_dt(tree, feature_names, target_class=1, min_samples=None, value_threshold=None):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    rules = []
    pred_strengths = []
    samples_per_rule = []
    features_per_rule = []
    used_features = set()
    
    def recurse(node, conditions, features_in_path):
        # if (tree_.feature[node] != _tree.TREE_UNDEFINED) and (tree_.n_node_samples[node] > min_samples):
        recurse_condition = (tree_.feature[node] != _tree.TREE_UNDEFINED)
        if min_samples is not None:
            recurse_condition = recurse_condition and (tree_.n_node_samples[node] > min_samples)
        if value_threshold is not None:
            values = tree_.value[node][0]
            recurse_condition = recurse_condition and (values[target_class].item() < value_threshold)

        if recurse_condition:  # not a leaf
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # left child (feature <= threshold)
            # recurse(tree_.children_left[node],
            #         conditions + [f"({name} <= {threshold:.4f})"],
            #         features_in_path | {name})
            recurse(tree_.children_left[node],
                    conditions + [f"(NOT {name})"],
                    features_in_path | {name})
            
            # right child (feature > threshold)
            # recurse(tree_.children_right[node],
            #         conditions + [f"({name} > {threshold:.4f})"],
            #         features_in_path | {name})
            recurse(tree_.children_right[node],
                    conditions + [f"({name})"],
                    features_in_path | {name})
        else:
            # Leaf node: check predicted class
            values = tree_.value[node][0]
            pred_class = values.argmax()
            if pred_class == target_class:
                rule = " AND ".join(conditions)
                rules.append(rule)
                pred_strengths.append(values[pred_class].item() / values.sum().item())
                samples_per_rule.append(tree_.n_node_samples[node].item())
                features_per_rule.append(features_in_path)
                used_features.update(features_in_path)
    
    recurse(0, [], set())
    return (rules, pred_strengths, samples_per_rule, features_per_rule), used_features

# %%
def extract_rules_features_from_reg_dt(
    num_layers,
    num_neurons, 
    reg_decision_trees,
    r2_scores,
    reg_feature_names,
    r2_threshold=0.7,
):
    dt_rules = defaultdict(dict)
    features_dict = defaultdict(dict)
    for layer in trange(num_layers):
        for neuron in range(num_neurons):
            if neuron not in reg_decision_trees[layer]:
                continue
            # reg_tree_model = reg_decision_trees[layer][neuron].tree
            # r2 = reg_decision_trees[layer][neuron].test_R2
            # reg_tree_model = reg_decision_trees[layer][custom_function_name]['decision_tree']['model'].estimators_[neuron]
            # r2 = reg_decision_trees[layer][custom_function_name]['decision_tree']['r2'][neuron].item()
            reg_tree_model = reg_decision_trees[layer][neuron]
            r2 = r2_scores[layer][neuron]

            # if r2 < r2_threshold:
            #     continue

            leaf_mask = reg_tree_model.tree_.children_right == -1
            leaf_values = reg_tree_model.tree_.value[leaf_mask].flatten()
            on_off_threshold = threshold_otsu(leaf_values)

            (rules, pred_act, samples_per_rule, features_per_rule), used_features = extract_and_rules_reg_dt(reg_tree_model, reg_feature_names, on_off_threshold=on_off_threshold)

            sorted_rules = sorted(
                zip(rules, pred_act, samples_per_rule, features_per_rule),
                key=lambda x: (x[2], x[1]),  # sort by samples_per_rule, then pred_strength
                reverse=True
            )

            filter_min_samples = reg_tree_model.tree_.n_node_samples[0].item() / 59 * .05
            filtered_rules = [(rule_infer(rule), strength, samples) for rule, strength, samples, features in sorted_rules if samples >= filter_min_samples]

            filtered_features = set()
            filtered_direct_features = set()
            for rule, strength, samples in filtered_rules:
                # print(f"Rule: {rule}\n\t(Strength: {strength:.2f}, Samples: {samples})")
                direct_feat_infered = set(rule.split(" AND "))
                feature_inferred = {feat.split("(")[-1].split(")")[0].split(" ")[-1] for feat in direct_feat_infered}
                
                filtered_features.update(feature_inferred)
                filtered_direct_features.update(direct_feat_infered)

            dt_rules[layer][neuron] = {
                "dt_rules": filtered_rules,
                "dt_filtered_features": filtered_features,
                "dt_filtered_directional_features": filtered_direct_features,
                "dt_r2": r2,
            }
    
    return dt_rules

# %%
def extract_and_rules_reg_dt(
    tree,
    feature_names,
    on_off_threshold=1,
    min_samples=None,
    # operate_thresholds = {"<=": 1, ">":-1}, 
):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    rules = []
    pred_act = []
    samples_per_rule = []
    features_per_rule = []
    used_features = set()
    
    def recurse(node, conditions, features_in_path):
        # if (tree_.feature[node] != _tree.TREE_UNDEFINED) and (tree_.n_node_samples[node] > min_samples):
        recurse_condition = (tree_.feature[node] != _tree.TREE_UNDEFINED)
        if min_samples is not None:
            recurse_condition = recurse_condition and (tree_.n_node_samples[node] > min_samples)
        # if value_threshold is not None:
        #     values = tree_.value[node][0][0]
        #     recurse_condition = recurse_condition and (values[target_class].item() < value_threshold)

        if recurse_condition:  # not a leaf
            name = feature_name[node]
            # threshold = tree_.threshold[node]
            
            # left child (feature <= threshold)
            # recurse(tree_.children_left[node],
            #         conditions + [f"({name} <= {threshold:.4f})"],
            #         features_in_path | {name})
            recurse(tree_.children_left[node],
                    conditions + [f"(NOT {name})"],
                    features_in_path | {name})
            
            # right child (feature > threshold)
            # recurse(tree_.children_right[node],
            #         conditions + [f"({name} > {threshold:.4f})"],
            #         features_in_path | {name})
            recurse(tree_.children_right[node],
                    conditions + [f"({name})"],
                    features_in_path | {name})
        else:
            # Leaf node: check predicted class
            values = tree_.value[node][0][0].item()
            if values > on_off_threshold:
                rule = " AND ".join(conditions)
                rules.append(rule)
                pred_act.append(values)
                samples_per_rule.append(tree_.n_node_samples[node].item())
                features_per_rule.append(features_in_path)
                used_features.update(features_in_path)
    
    recurse(0, [], set())
    return (rules, pred_act, samples_per_rule, features_per_rule), used_features

# %%
def extract_probe_features(matrices, k=2):
    matrices_mean = matrices.mean().item()
    matrices_std = matrices.std().item()

    filtered_feature_names = []
    directional_feature_names = []
    for row in range(8):
        for col in range(8):
            square = chr(ord('A') + row) + str(col)
            mine_weight = matrices[0, row, col].item()
            empty_weight = matrices[1, row, col].item()
            theirs_weight = matrices[2, row, col].item()
            flipped_weight = matrices[3, row, col].item()
            just_played_weight = matrices[4, row, col].item()

            occupied = 0
            if mine_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_mine")
                directional_feature_names.append(f"({square}_mine)")
                # occupied = 1

            if mine_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_mine")
                directional_feature_names.append(f"(NOT {square}_mine)")

            if theirs_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_theirs")
                directional_feature_names.append(f"({square}_theirs)")
                # occupied = 1
            
            if theirs_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_theirs")
                directional_feature_names.append(f"(NOT {square}_theirs)")

            if empty_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_empty")
                directional_feature_names.append(f"({square}_empty)")
            
            if empty_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_empty")
                directional_feature_names.append(f"(NOT {square}_empty)")
                # occupied = 1
            
            if flipped_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_flipped")
                directional_feature_names.append(f"({square}_flipped)")
                # occupied = 1
            
            if flipped_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_flipped")
                directional_feature_names.append(f"(NOT {square}_flipped)")

            if just_played_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_just_played")
                directional_feature_names.append(f"({square}_just_played)")
                # occupied = 1
            
            if just_played_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_just_played")
                directional_feature_names.append(f"(NOT {square}_just_played)")

    return filtered_feature_names, directional_feature_names

# %%
def extract_fold_probe_features(matrices, k=2):
    matrices_mean = matrices.mean().item()
    matrices_std = matrices.std().item()

    filtered_feature_names = []
    directional_feature_names = []
    for row in range(8):
        for col in range(8):
            square = chr(ord('A') + row) + str(col)
            blank_weight = matrices[0, row, col].item()
            my_weight = matrices[1, row, col].item()
            flipped_weight = matrices[2, row, col].item()
            just_played_weight = matrices[3, row, col].item()

            # mine_weight = matrices[0, row, col].item()
            # empty_weight = matrices[1, row, col].item()
            # theirs_weight = matrices[2, row, col].item()
            # flipped_weight = matrices[3, row, col].item()
            # just_played_weight = matrices[4, row, col].item()

            occupied = 0
            if blank_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_blank")
                directional_feature_names.append(f"({square}_blank)")
                # occupied = 1

            if blank_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_blank")
                directional_feature_names.append(f"(NOT {square}_blank)")
                # occupied = 1

            if my_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_mine-theirs")
                directional_feature_names.append(f"({square}_mine-theirs)")
                # occupied = 1
            
            if my_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_mine-theirs")
                directional_feature_names.append(f"(NOT {square}_mine-theirs)")
                # occupied = 1
            
            if flipped_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_flipped")
                directional_feature_names.append(f"({square}_flipped)")
                # occupied = 1
            
            if flipped_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_flipped")
                directional_feature_names.append(f"(NOT {square}_flipped)")

            if just_played_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_just_played")
                directional_feature_names.append(f"({square}_just_played)")
                # occupied = 1
            
            if just_played_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_just_played")
                directional_feature_names.append(f"(NOT {square}_just_played)")

    return filtered_feature_names, directional_feature_names

# %%
def infer_positive_from_negations(features, group, values):
    """
    If all but one value in a group are negated, infer the positive for the last one.
    """
    feats = set(features)
    negs = {v for v in values if f"(NOT {group}_{v})" in feats}
    remaining = set(values) - negs

    if len(remaining) == 1 and len(negs) == len(values) - 1:
        # Replace negations with the positive
        inferred = f"({group}_{remaining.pop()})"
        feats = (feats - {f"(NOT {group}_{v})" for v in negs}) | {inferred}

    return feats

def remove_negations_if_positive_present(features, group, values):
    """
    If a positive feature is present, remove all negations of the same group.
    """
    feats = set(features)
    positives = [f"({group}_{v})" for v in values if f"({group}_{v})" in feats]
    
    if positives:
        # Remove all negations if any positive is present
        feats = feats - {f"(NOT {group}_{v})" for v in values}
    
    return feats

def extract_squares(directional_features):
    squares = set()
    for feat in directional_features:
        squares.add(feat.split("(")[-1].split(" ")[-1].split("_")[0])
    return squares

def direct_feature_infer(directional_features):
    squares = extract_squares(directional_features)
    for square in squares:
        directional_features = infer_positive_from_negations(directional_features, square, ["mine", "theirs", "empty"])
        directional_features = remove_negations_if_positive_present(directional_features, square, ["mine", "theirs", "empty"])
    # filtered = set()
    # for feat in features:
    #     if feat.startswith("NOT "):
    #         continue
    #     if feat.endswith("_flipped") or feat.endswith("_just_played"):
    #         continue
    #     filtered.add(feat)
    return directional_features

def rule_infer(rule):
    directional_features = set(rule.split(" AND "))
    direct_feat_inferred = direct_feature_infer(directional_features)
    rule_inferred = " AND ".join(sorted(direct_feat_inferred))
    return rule_inferred

# %%
def set_overlap_metrics(set1, set2):
    """
    Compute pure set overlap metrics between two sets of features.

    Parameters:
        set1, set2 : iterable of features (list, set, etc.)

    Returns:
        dict with Jaccard index and Overlap coefficient
    """
    set1 = set(set1)
    set2 = set(set2)
    
    intersection = set1 & set2
    union = set1 | set2
    
    jaccard = len(intersection) / len(union) if union else 1.0
    overlap_coef = len(intersection) / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 1.0
    set1_in_set2 = len(intersection) / len(set2) if len(set2) > 0 else 1.0
    set2_in_set1 = len(intersection) / len(set1) if len(set1) > 0 else 1.0
    
    return {
        "jaccard_index": jaccard,
        "overlap_coefficient": overlap_coef,
        "set1_in_set2": set1_in_set2,
        "set2_in_set1": set2_in_set1,
    }

# %%
def aggregate_scores(d, score_key="f1"):
    """
    Aggregates scores by layer across neurons.
    Assumes structure: dict[layer][neuron][info] = score
    """
    layer_scores = {}
    for layer, neurons in d.items():
        scores = []
        if isinstance(neurons, list):
            for neuron in neurons:
                score = neuron.get(score_key)
                if score is not None:
                    scores.append(score)
        elif isinstance(neurons, dict):
            for _, neuron in neurons.items():
                score = neuron.get(score_key)
                if score is not None:
                    scores.append(score)
        layer_scores[layer] = scores
    return layer_scores
