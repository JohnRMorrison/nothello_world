"""New Squares Experiment: Coherent vs Incoherent Spatial Addition

Tests whether OthelloGPT learns new squares faster when they fit the
existing board geometry vs when they're spatially arbitrary.

Condition 0: Coherent — 8 new squares as a 9th row (A9-H9)
Condition 1: Incoherent — 8 new squares with random neighbors

Each new square has 5 flanking rules. The model's vocab is expanded
from 61 to 69 (8 new token embeddings, randomly initialized).

Usage:
    python new_squares_experiment.py --condition-id 0 --output-dir experiments/new_squares
"""

import argparse
import json
import os
import sys
import time
import pickle
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
from rules_960 import enumerate_flanking_patterns, N_MOVES, VALID_MOVES
from transfer_utils import load_model
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig
from torch.utils.data import DataLoader
from data.othello import OthelloBoardState

# Standard Othello has 60 playable cells (0-63 minus center 4)
# New squares are cells 64-71
CENTER_CELLS = {27, 28, 35, 36}
NEW_SQUARE_IDS = list(range(64, 72))  # 8 new squares
N_NEW = 8
RULES_PER_SQUARE = 5

EVAL_SCHEDULE = [0, 5, 25, 50, 100, 200, 500, 1000, 2000, 5000,
                 10000, 20000, 50000, 100000]


# ============================================================================
# Rule generation for new squares
# ============================================================================

def create_coherent_rules(rng):
    """Create rules for 8 new squares as a 9th row.

    New squares 64-71 correspond to A9-H9, below row 8 (cells 56-63).
    Each gets 5 rules using standard Othello geometry:
      - rightward along row 9
      - leftward along row 9
      - upward into existing board
      - diagonal up-right
      - diagonal up-left
    """
    rules = []

    for i in range(N_NEW):
        new_cell = 64 + i  # A9=64, B9=65, ..., H9=71
        col = i  # column 0-7
        above_cell = 56 + col  # cell in row 8 directly above

        # Generate candidate rules using local geometry
        candidates = []

        # 1. Upward (into existing board): new_cell -> row8 -> row7 -> ...
        chain = []
        for row in range(7, -1, -1):  # rows 7,6,5,4,3,2,1,0
            cell = row * 8 + col
            if cell in CENTER_CELLS:
                break
            chain.append(cell)
            if len(chain) >= 2:  # need at least 1 opponent + 1 terminal
                candidates.append({
                    'target': new_cell,
                    'opponents': list(chain[:-1]),
                    'terminal': chain[-1],
                    'direction': 'up',
                })

        # 2. Rightward along row 9
        chain = []
        for j in range(col + 1, min(col + 7, 8)):
            chain.append(64 + j)
            if len(chain) >= 2:
                candidates.append({
                    'target': new_cell,
                    'opponents': list(chain[:-1]),
                    'terminal': chain[-1],
                    'direction': 'right_row9',
                })

        # 3. Leftward along row 9
        chain = []
        for j in range(col - 1, max(col - 7, -1), -1):
            chain.append(64 + j)
            if len(chain) >= 2:
                candidates.append({
                    'target': new_cell,
                    'opponents': list(chain[:-1]),
                    'terminal': chain[-1],
                    'direction': 'left_row9',
                })

        # 4. Diagonal up-right
        chain = []
        r, c = 7, col + 1  # start from row 8, one col right
        while r >= 0 and c < 8:
            cell = r * 8 + c
            if cell in CENTER_CELLS:
                break
            chain.append(cell)
            if len(chain) >= 2:
                candidates.append({
                    'target': new_cell,
                    'opponents': list(chain[:-1]),
                    'terminal': chain[-1],
                    'direction': 'diag_up_right',
                })
            r -= 1
            c += 1

        # 5. Diagonal up-left
        chain = []
        r, c = 7, col - 1
        while r >= 0 and c >= 0:
            cell = r * 8 + c
            if cell in CENTER_CELLS:
                break
            chain.append(cell)
            if len(chain) >= 2:
                candidates.append({
                    'target': new_cell,
                    'opponents': list(chain[:-1]),
                    'terminal': chain[-1],
                    'direction': 'diag_up_left',
                })
            r -= 1
            c -= 1

        # Select 5 rules, preferring diverse directions
        selected = []
        dirs_used = set()
        # First pass: one per direction
        for cand in candidates:
            if cand['direction'] not in dirs_used and len(selected) < RULES_PER_SQUARE:
                selected.append(cand)
                dirs_used.add(cand['direction'])
        # Fill remaining from any direction
        for cand in candidates:
            if len(selected) >= RULES_PER_SQUARE:
                break
            if cand not in selected:
                selected.append(cand)

        # If we don't have enough (edge squares), pad with shorter chains
        while len(selected) < RULES_PER_SQUARE and candidates:
            selected.append(rng.choice(candidates))

        rules.extend(selected[:RULES_PER_SQUARE])

    return rules


def create_incoherent_rules(rng):
    """Create rules for 8 new squares with random neighbors.

    Each new square gets 5 rules where opponents and terminal are
    randomly chosen existing cells (0-63, excluding center).
    Chain lengths match the distribution from the coherent version.
    """
    playable = [c for c in range(64) if c not in CENTER_CELLS]

    # Get chain length distribution from coherent rules
    coherent_rules = create_coherent_rules(np.random.RandomState(0))
    chain_lengths = [len(r['opponents']) + 1 for r in coherent_rules]  # +1 for terminal

    rules = []
    for i in range(N_NEW):
        new_cell = 64 + i

        for rule_idx in range(RULES_PER_SQUARE):
            # Match chain length from coherent
            length_idx = i * RULES_PER_SQUARE + rule_idx
            if length_idx < len(chain_lengths):
                chain_len = chain_lengths[length_idx]
            else:
                chain_len = rng.choice(chain_lengths)

            # Random cells for opponents + terminal
            n_opp = max(1, chain_len - 1)
            cells = rng.choice(playable, size=n_opp + 1, replace=False)

            rules.append({
                'target': new_cell,
                'opponents': cells[:-1].tolist(),
                'terminal': int(cells[-1]),
                'direction': 'random',
            })

    return rules


# ============================================================================
# Extended game engine (supports cells 64-71)
# ============================================================================

def generate_game_with_new_squares(new_rules, rng, save_legal=False):
    """Generate a game on the standard 8x8 board + 8 new squares.

    The standard board uses OthelloBoardState for the original 64 cells.
    New squares (64-71) are tracked separately.
    """
    board = OthelloBoardState()

    # Track new squares: 0=empty, 1=black, -1=white
    new_state = np.zeros(N_NEW, dtype=np.int32)

    # Precompute rule arrays for new squares
    new_targets = np.array([r['target'] for r in new_rules], dtype=np.int32)
    new_terminals = np.array([r['terminal'] for r in new_rules], dtype=np.int32)
    max_opp = max(len(r['opponents']) for r in new_rules) if new_rules else 1
    new_opp_cells = np.zeros((len(new_rules), max_opp), dtype=np.int32)
    new_opp_lens = np.array([len(r['opponents']) for r in new_rules], dtype=np.int32)
    for i, r in enumerate(new_rules):
        for j, o in enumerate(r['opponents']):
            new_opp_cells[i, j] = o
    new_opp_mask = np.arange(max_opp)[None, :] < new_opp_lens[:, None]

    moves = []
    legal_per_turn = [] if save_legal else None

    for turn in range(68):  # slightly more turns possible with new squares
        flat = board.state.flatten()  # 64 cells
        is_black = (board.next_hand_color == 1)
        my_val = 1 if is_black else -1
        opp_val = -my_val

        # Standard legal moves
        std_legal = board.get_valid_moves()

        # Check new square rules
        # Build extended flat array (64 standard + 8 new)
        ext_flat = np.concatenate([flat, new_state])

        new_legal = set()
        for i, rule in enumerate(new_rules):
            target = rule['target']
            # Target must be empty
            if target >= 64:
                if new_state[target - 64] != 0:
                    continue
            else:
                if flat[target] != 0:
                    continue

            # Check opponents
            opp_ok = True
            for opp_cell in rule['opponents']:
                if opp_cell >= 64:
                    val = new_state[opp_cell - 64]
                else:
                    val = flat[opp_cell]
                if val != opp_val:
                    opp_ok = False
                    break
            if not opp_ok:
                continue

            # Check terminal
            term_cell = rule['terminal']
            if term_cell >= 64:
                term_val = new_state[term_cell - 64]
            else:
                term_val = flat[term_cell]
            if term_val != my_val:
                continue

            new_legal.add(target)

        # Combine legal moves
        all_legal = list(std_legal) + list(new_legal)

        if not all_legal:
            # Pass or end game
            empty_std = np.where(flat == 0)[0]
            empty_std = [c for c in empty_std if c not in CENTER_CELLS]
            empty_new = [64 + j for j in range(N_NEW) if new_state[j] == 0]
            all_empty = empty_std + empty_new
            if not all_empty:
                break
            # Place without flips
            board_pos = int(rng.choice(all_empty))
        else:
            if save_legal:
                legal_per_turn.append(all_legal)
            board_pos = int(rng.choice(all_legal))

        # Execute move
        if board_pos >= 64:
            # Move on new square
            new_state[board_pos - 64] = my_val
            board.next_hand_color *= -1  # switch turns
        else:
            # Standard move
            if board.tentative_move(board_pos) != 0:
                board.update([board_pos])
            else:
                # Place without flip if not a valid standard move
                r, c = board_pos // 8, board_pos % 8
                board.state[r, c] = my_val
                board.next_hand_color *= -1

        moves.append(board_pos)

        if len(moves) >= 68:
            break

    if save_legal:
        return moves, legal_per_turn
    return moves


def generate_games_new_squares(new_rules, num_games, rng, save_legal=False):
    """Generate multiple games with new squares."""
    all_games = []
    all_legal = [] if save_legal else None

    for i in range(num_games):
        if save_legal:
            moves, legal = generate_game_with_new_squares(new_rules, rng, save_legal=True)
            all_games.append(moves)
            all_legal.append(legal)
        else:
            moves = generate_game_with_new_squares(new_rules, rng)
            all_games.append(moves)

        if (i + 1) % 10000 == 0:
            print(f"  Generated {i+1}/{num_games} games...", flush=True)

    return all_games, all_legal


# ============================================================================
# Model expansion
# ============================================================================

def expand_model_vocab(model, old_vocab=61, new_vocab=69, device='cpu'):
    """Expand model's embedding and output layers to accommodate new tokens.

    New embeddings are randomly initialized (std=0.02, matching GPT init).
    Existing weights are preserved exactly.
    """
    n_embd = model.tok_emb.weight.shape[1]

    # Expand token embedding
    new_tok_emb = nn.Embedding(new_vocab, n_embd).to(device)
    new_tok_emb.weight.data[:old_vocab] = model.tok_emb.weight.data
    new_tok_emb.weight.data[old_vocab:].normal_(mean=0.0, std=0.02)
    model.tok_emb = new_tok_emb

    # Expand output head
    new_head = nn.Linear(n_embd, new_vocab, bias=False).to(device)
    new_head.weight.data[:old_vocab] = model.head.weight.data
    new_head.weight.data[old_vocab:].normal_(mean=0.0, std=0.02)
    model.head = new_head

    return model


# ============================================================================
# Extended CharDataset for new vocab
# ============================================================================

class ExtendedCharDataset(CharDataset):
    """CharDataset that handles new square IDs (64-71)."""

    def __init__(self, data, new_square_ids=None):
        # Build vocab including new squares
        all_vals = set()
        for game in data:
            all_vals.update(game)
        all_vals.add(-100)  # padding
        chars = sorted(list(all_vals))

        data_size = len(data)
        vocab_size = len(chars)
        max_len = max(len(game) for game in data)

        print(f'Dataset created has {data_size} sequences, {vocab_size} unique words.')

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = max_len
        self.block_size = max_len - 1
        self.vocab_size = vocab_size
        self.data = data


# ============================================================================
# Test set collection
# ============================================================================

def collect_new_square_test_set(games, legal_moves_per_game, new_square_ids,
                                 n_positions=5000, min_turn=4):
    """Collect test positions where new squares are legal (IL positions).

    Also collects LL positions (existing moves that remain legal).
    """
    test_IL = []  # positions where a new square is legal
    test_LL = []  # positions where standard moves are legal
    games_used_IL = set()
    games_used_LL = set()
    new_set = set(new_square_ids)

    for game_idx, (game, legals) in enumerate(zip(games, legal_moves_per_game)):
        if len(test_IL) >= n_positions and len(test_LL) >= n_positions:
            break

        for turn, legal in enumerate(legals):
            if turn < min_turn:
                continue

            prefix = game[:turn]
            new_legal = [m for m in legal if m in new_set]
            std_legal = [m for m in legal if m not in new_set]

            if new_legal and len(test_IL) < n_positions and game_idx not in games_used_IL:
                test_IL.append({
                    'game_prefix': list(prefix),
                    'target_cells': new_legal,
                    'all_legal': list(legal),
                    'move_idx': turn,
                    'n_corrupted_rules': len(new_legal),
                })
                games_used_IL.add(game_idx)

            if std_legal and len(test_LL) < n_positions and game_idx not in games_used_LL:
                test_LL.append({
                    'game_prefix': list(prefix),
                    'target_cells': std_legal,
                    'all_legal': list(legal),
                    'move_idx': turn,
                })
                games_used_LL.add(game_idx)

    return {'IL': test_IL, 'LL': test_LL}


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_on_test_sets(model, test_sets, dataset, device):
    """Evaluate model on IL and LL test positions."""
    model.eval()
    results = {}

    with torch.no_grad():
        for set_name, positions in test_sets.items():
            if not positions:
                results[f'{set_name}_prob'] = 0.0
                results[f'{set_name}_acc'] = 0.0
                continue

            total_prob = 0.0
            total_acc = 0.0
            n = 0

            for pos in positions:
                prefix = pos['game_prefix']
                target_cells = pos['target_cells']

                if len(prefix) < 1:
                    continue

                # Tokenize
                tokens = []
                for m in prefix:
                    if m in dataset.stoi:
                        tokens.append(dataset.stoi[m])
                    else:
                        continue  # skip unknown tokens

                if len(tokens) < 1:
                    continue

                x = torch.tensor([tokens], dtype=torch.long, device=device)
                logits, _ = model(x)
                probs = torch.softmax(logits[0, -1, :], dim=-1)

                # Probability on target cells
                prob_sum = 0.0
                for cell in target_cells:
                    if cell in dataset.stoi:
                        token_idx = dataset.stoi[cell]
                        if token_idx < probs.shape[0]:
                            prob_sum += probs[token_idx].item()

                total_prob += prob_sum

                # Top-1 accuracy: is the argmax in the target set?
                top1 = probs.argmax().item()
                top1_cell = dataset.itos.get(top1, -1)
                total_acc += 1.0 if top1_cell in set(target_cells) else 0.0

                n += 1

            results[f'{set_name}_prob'] = total_prob / max(n, 1)
            results[f'{set_name}_acc'] = total_acc / max(n, 1)

    return results


def evaluate_standard_lpm(model, std_loader, std_mask, device):
    """Evaluate LPM on standard Othello games."""
    model.eval()
    total_lpm = 0.0
    total_acc = 0.0
    n = 0

    with torch.no_grad():
        for x, y in std_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=-1)  # (B, T, V)

            B, T, V = probs.shape
            # Only evaluate up to vocab 61 (standard moves)
            V_std = min(V, std_mask.shape[1])

            for b in range(B):
                for t in range(T):
                    if y[b, t] == 0:  # padding
                        continue
                    mask_row = std_mask[n % std_mask.shape[0]]
                    lpm = probs[b, t, :V_std][mask_row[:V_std] > 0].sum().item()
                    top1 = probs[b, t, :V_std].argmax().item()
                    acc = 1.0 if mask_row[top1] > 0 else 0.0
                    total_lpm += lpm
                    total_acc += acc
                    n += 1

    return total_lpm / max(n, 1), total_acc / max(n, 1)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-id", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="experiments/new_squares")
    parser.add_argument("--n-games", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    # Condition mapping
    # 0: coherent (9th row)
    # 1: incoherent (random neighbors)
    conditions = ['coherent', 'incoherent']
    cond_name = conditions[args.condition_id]

    print(f"Condition {args.condition_id}: {cond_name}", flush=True)
    print(f"Started at: {time.strftime('%c')}", flush=True)

    # Create rules
    if cond_name == 'coherent':
        new_rules = create_coherent_rules(rng)
    else:
        new_rules = create_incoherent_rules(rng)

    print(f"Created {len(new_rules)} rules for {N_NEW} new squares", flush=True)
    for i, rule in enumerate(new_rules):
        target = rule['target']
        opps = rule['opponents']
        term = rule['terminal']
        print(f"  Rule {i}: target={target} opps={opps} term={term} dir={rule['direction']}")

    # Generate games
    games_dir = os.path.join(args.output_dir, 'games', f'cond_{args.condition_id:03d}')
    os.makedirs(games_dir, exist_ok=True)

    games_path = os.path.join(games_dir, 'train_games.pickle')
    legal_path = os.path.join(games_dir, 'train_legal.pickle')

    if os.path.exists(games_path):
        print(f"Loading existing games from {games_dir}...", flush=True)
        with open(games_path, 'rb') as f:
            games = pickle.load(f)
        with open(legal_path, 'rb') as f:
            legal_moves = pickle.load(f)
        print(f"  Loaded {len(games)} games", flush=True)
    else:
        print(f"Generating {args.n_games} games...", flush=True)
        games, legal_moves = generate_games_new_squares(
            new_rules, args.n_games, rng, save_legal=True)

        # Filter short games
        filtered = [(g, l) for g, l in zip(games, legal_moves) if len(g) >= 5]
        games = [g for g, _ in filtered]
        legal_moves = [l for _, l in filtered]
        print(f"  {len(games)} games after filtering", flush=True)

        with open(games_path, 'wb') as f:
            pickle.dump(games, f)
        with open(legal_path, 'wb') as f:
            pickle.dump(legal_moves, f)

    # Collect test positions
    print("Collecting test positions...", flush=True)
    test_sets = collect_new_square_test_set(
        games, legal_moves, NEW_SQUARE_IDS, n_positions=5000)
    print(f"  IL: {len(test_sets.get('IL', []))} positions", flush=True)
    print(f"  LL: {len(test_sets.get('LL', []))} positions", flush=True)

    # Load and expand model
    print("Loading model...", flush=True)
    model, dataset, device = load_model()
    old_vocab = model.tok_emb.weight.shape[0]

    # Truncate games to 60 moves (model block_size = 59, needs seq length 60)
    MAX_MOVES = 60
    games = [g[:MAX_MOVES] for g in games]
    legal_moves = [l[:MAX_MOVES] for l in legal_moves]

    # Create extended dataset to get proper stoi/itos
    n_train = int(len(games) * 0.95)
    train_games = games[:n_train]

    train_ds = ExtendedCharDataset(train_games)
    new_vocab = train_ds.vocab_size

    print(f"Expanding model vocab: {old_vocab} -> {new_vocab}", flush=True)
    model = expand_model_vocab(model, old_vocab, new_vocab, device)

    # Build standard LPM test
    print("Building standard test set...", flush=True)
    from transfer_utils import build_legal_mask, load_shard_games
    std_games_raw = load_shard_games(0, games_per_shard=2000)
    std_legal_raw = []
    for g in std_games_raw:
        b = OthelloBoardState()
        lm = []
        for move in g:
            lm.append(b.get_valid_moves())
            b.umpire(move)
        std_legal_raw.append(lm)

    # Use the TRAINING dataset's stoi for standard games too
    std_ds = ExtendedCharDataset(std_games_raw)
    # Remap stoi to match training dataset
    std_ds.stoi = train_ds.stoi
    std_ds.itos = train_ds.itos
    std_ds.vocab_size = train_ds.vocab_size

    std_mask = build_legal_mask(std_games_raw, std_legal_raw,
                                 train_ds.stoi, train_ds.block_size, train_ds.vocab_size)
    std_loader = DataLoader(std_ds, batch_size=64, shuffle=False)

    # Build corrupted LPM test
    cor_test_games = games[n_train:n_train + 2000]
    cor_test_legal = legal_moves[n_train:n_train + 2000]
    cor_ds = ExtendedCharDataset(cor_test_games)
    cor_ds.stoi = train_ds.stoi
    cor_ds.itos = train_ds.itos
    cor_ds.vocab_size = train_ds.vocab_size
    cor_mask = build_legal_mask(cor_test_games, cor_test_legal,
                                 train_ds.stoi, train_ds.block_size, train_ds.vocab_size)
    cor_loader = DataLoader(cor_ds, batch_size=64, shuffle=False)

    # Training
    print("Training...", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    total_steps = len(train_loader)
    eval_at = set(EVAL_SCHEDULE + [total_steps - 1])

    results = {
        'condition_id': args.condition_id,
        'condition_name': cond_name,
        'n_new_squares': N_NEW,
        'rules_per_square': RULES_PER_SQUARE,
        'n_rules': len(new_rules),
        'rules': [{'target': r['target'], 'opponents': r['opponents'],
                    'terminal': r['terminal'], 'direction': r['direction']}
                   for r in new_rules],
        'n_train_games': len(train_games),
        'total_steps': total_steps,
        'lr': args.lr,
        'bs': args.bs,
        'seed': args.seed,
        'eval_steps': [],
        'IL_prob': [], 'IL_acc': [],
        'LL_prob': [], 'LL_acc': [],
        'STD_prob': [], 'COR_prob': [],
        'std_top': [], 'cor_top': [],
    }

    t0 = time.time()

    def do_eval(step):
        model.eval()

        # IL/LL metrics
        metrics = evaluate_on_test_sets(model, test_sets, train_ds, device)

        # Standard LPM
        std_prob, std_top = evaluate_standard_lpm(model, std_loader, std_mask, device)

        # Corrupted LPM
        cor_prob, cor_top = evaluate_standard_lpm(model, cor_loader, cor_mask, device)

        elapsed = time.time() - t0

        results['eval_steps'].append(step)
        results['IL_prob'].append(metrics.get('IL_prob', 0.0))
        results['IL_acc'].append(metrics.get('IL_acc', 0.0))
        results['LL_prob'].append(metrics.get('LL_prob', 0.0))
        results['LL_acc'].append(metrics.get('LL_acc', 0.0))
        results['STD_prob'].append(std_prob)
        results['std_top'].append(std_top)
        results['COR_prob'].append(cor_prob)
        results['cor_top'].append(cor_top)

        print(f"  Step {step}: IL_prob={metrics.get('IL_prob',0):.4f} "
              f"LL_prob={metrics.get('LL_prob',0):.4f} "
              f"STD_prob={std_prob:.4f} COR_prob={cor_prob:.4f} "
              f"elapsed={elapsed:.0f}s", flush=True)

        model.train()

    # Initial eval
    do_eval(0)

    model.train()
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) in eval_at:
            do_eval(step + 1)

    # Save
    results['elapsed_seconds'] = time.time() - t0

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'cond_{args.condition_id:03d}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}", flush=True)

    # Also save rules
    rules_path = os.path.join(games_dir, 'rules.json')
    with open(rules_path, 'w') as f:
        json.dump(new_rules, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else x)


if __name__ == "__main__":
    main()
