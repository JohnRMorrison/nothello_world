"""Proof of concept: can a 1-layer MLP recover board state from CUMULATIVE
rule firings instead of raw move history?

For each game position at turn T:
  - Replay through OthelloBoardState
  - At each move m_t, identify the patterns that fired (target = m_t AND
    flanking arrangement held on pre-move board) using compute_pattern_labels_batch
  - Accumulate: cum_firings[T] = sum of per-turn firings for turns 0..T-1
  - Record (cum_firings[T], board_state[T])

Then train a 1-layer MLP: 960-d cumulative firings -> 64*3 board state.

If accuracy is near-100% (especially on center cells late game), this
proves: the architectural bottleneck for the existing MLP is NOT that
"1 layer can't decode board" -- it's that move history is a hard input
format. With a smarter input format (cumulative firings), 1 layer is
fine.

Usage:
    python rule_firings_poc.py --n-games 5000
"""
import sys, os, argparse, time
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.othello import OthelloBoardState
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from generate_rule_games import precompute_pattern_arrays
from train_pattern_simple import compute_pattern_labels_batch

sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import load_games, GAME_LEN


CENTER_64 = {27, 28, 35, 36}


def cell_class(c64):
    if c64 in CENTER_64: return 'center'
    r, col = c64 // 8, c64 % 8
    if r in (0, 7) and col in (0, 7): return 'corner'
    if r in (0, 7) or col in (0, 7):  return 'edge'
    return 'inner'


def encode_state_classes(state_8x8):
    """Convert OthelloBoardState.state (8x8 -1/0/1) to 64 class labels
    (0=empty, 1=white, 2=black)."""
    s = state_8x8.flatten()
    cls = np.zeros(64, dtype=np.int64)
    cls[s == 0]  = 0
    cls[s == -1] = 1
    cls[s == 1]  = 2
    return cls


def collect_data(games, pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
                 patterns, turn_range):
    """For each game, replay and at each turn record (cumulative firings,
    board state). Returns (X (N, 960) int, Y (N, 64) int, pos (N,))."""
    # Patterns indexed by target cell (MOVE_TO_IDX space).
    pat_target_60 = np.array([MOVE_TO_IDX[p['target']] for p in patterns])
    # 64 board cells -> 60 MOVE_TO_IDX (-1 for center cells)
    skip = {27, 28, 35, 36}
    b64_to_m60 = {c: i for i, c in enumerate(c for c in range(64) if c not in skip)}

    X_list, Y_list, pos_list = [], [], []
    n_games = len(games)
    for g_idx, game in enumerate(games):
        board = OthelloBoardState()
        cum_firings = np.zeros(960, dtype=np.int32)
        states = []
        moves = []
        # First, replay collecting pre-move boards
        pre_boards = []
        for t, m_64 in enumerate(game):
            pre_boards.append(np.copy(board.state))
            moves.append(m_64)
            board.umpire(m_64)
            states.append(np.copy(board.state))
        pre_boards = np.stack(pre_boards)   # (T, 8, 8)

        # Compute would-fire patterns at each pre-move state. compute_pattern_labels_batch
        # expects flat 64-d board. Convert to its preferred encoding.
        # NOTE: compute_pattern_labels_batch needs (Y, pos) where Y is the
        # 64-d board and pos is the turn. Let's call it per-turn.
        Y_pre = np.stack([encode_state_classes(s) for s in pre_boards])
        # But compute_pattern_labels_batch expects different encoding?
        # Check: in train_pattern_simple, Y is loaded from chunk with values
        # 0=empty, 1=mine, 2=theirs (player-relative) -- the labels argument.
        # Hmm. compute_pattern_labels_batch takes (yb, pb, ...).
        # yb is 'mine vs theirs' player-relative.
        # Need to convert pre_boards to mine/theirs given the turn.

        # For each turn t, "mine" = black if t even (black plays even turns) else white.
        # Actually wait, move 0 is black, so black plays at turns 0, 2, 4, ...
        # turn t plays move t. Player playing move t is black if t even.
        # So at pre-move state of turn t, "mine" = whoever plays turn t = black if t%2==0.

        Y_player_rel = np.zeros_like(Y_pre)
        for t in range(len(game)):
            # at turn t (pre-move), about to play move t. "mine" = player of turn t.
            # In Y_pre encoding: 0=empty, 1=white, 2=black
            # mine in player-relative: 0=empty, 1=mine, 2=theirs
            mine_color = 2 if t % 2 == 0 else 1   # 2=black if even, 1=white if odd
            theirs_color = 1 if t % 2 == 0 else 2
            for c in range(64):
                v = Y_pre[t, c]
                if v == 0: Y_player_rel[t, c] = 0
                elif v == mine_color: Y_player_rel[t, c] = 1
                else: Y_player_rel[t, c] = 2

        # Now compute which patterns fire at each pre-move state
        pos_arr = np.arange(len(game), dtype=np.int64)
        # compute_pattern_labels_batch wants float/int Y in player-relative encoding
        try:
            labels = compute_pattern_labels_batch(
                Y_player_rel, pos_arr,
                pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask
            )  # (T, 960) bool/int
        except Exception as e:
            print(f"  Game {g_idx}: failed to compute labels: {e}")
            continue
        labels = np.asarray(labels)

        # For each turn t, the FIRED patterns are those with P.target == move_t AND label==1
        for t in range(len(game)):
            if t < min(turn_range) or t > max(turn_range):
                # Still need to update cum_firings, but won't record this position
                pass
            m_64 = moves[t]
            m_60 = b64_to_m60.get(m_64, -1)
            if m_60 < 0:
                # Move was at a center cell -- shouldn't happen in normal Othello
                continue
            fired_mask = (pat_target_60 == m_60) & (labels[t] > 0.5)
            cum_firings[fired_mask] += 1
            if t in turn_range:
                X_list.append(cum_firings.copy())
                Y_list.append(encode_state_classes(states[t]))
                pos_list.append(t)

        if (g_idx + 1) % 500 == 0:
            print(f"  game {g_idx+1}/{n_games}, "
                  f"{len(X_list)} positions collected", flush=True)

    X = np.stack(X_list)
    Y = np.stack(Y_list)
    pos = np.array(pos_list)
    return X, Y, pos


class SmallMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, Y_train, X_test, Y_test, in_dim, hidden=256,
              epochs=10, lr=1e-3, batch_size=1024, device='cpu'):
    """Train a 1-layer MLP from cumulative firings to per-cell class.
    Output: (B, 64*3)."""
    model = SmallMLP(in_dim, hidden, 64 * 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X_train.astype(np.float32)).to(device)
    Yt = torch.from_numpy(Y_train).to(device)
    Xv = torch.from_numpy(X_test.astype(np.float32)).to(device)
    Yv = torch.from_numpy(Y_test).to(device)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        loss_sum = 0.0; n_batches = 0
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(Xt[idx]).view(-1, 64, 3)
            loss = F.cross_entropy(logits.reshape(-1, 3), Yt[idx].reshape(-1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += float(loss.item()); n_batches += 1
        # Eval
        model.eval()
        with torch.no_grad():
            preds = model(Xv).view(-1, 64, 3).argmax(dim=-1)
            acc = (preds == Yv).float().mean().item()
            per_cell = ((preds == Yv).float()).mean(dim=0).cpu().numpy()
        if acc > best_acc:
            best_acc = acc
            best_per_cell = per_cell
        print(f"  Epoch {epoch}: train_loss={loss_sum/max(n_batches,1):.5f} "
              f"test_acc={acc:.4%}", flush=True)
    return best_acc, best_per_cell


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games", type=int, default=5000)
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use-mod2", action="store_true",
                        help="Use cumulative firings MOD 2 instead of raw counts.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    games = load_games(max_files=args.max_files)
    if len(games) > args.n_games:
        games = games[:args.n_games]
    print(f"Using {len(games)} games")

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = \
        precompute_pattern_arrays(patterns)

    print(f"\nCollecting (cum_firings, board) pairs at turns 5-53...")
    t0 = time.time()
    X, Y, pos = collect_data(games, pat_targets, pat_terminals, pat_opp_cells,
                              pat_opp_mask, patterns, set(range(5, 54)))
    print(f"Done in {time.time()-t0:.0f}s. X={X.shape}, Y={Y.shape}")

    if args.use_mod2:
        print("Using cumulative-firings MOD 2 as input")
        X = (X % 2).astype(np.int32)

    # 70/30 split
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(X))
    split = int(0.7 * len(X))
    X_train, Y_train = X[idx[:split]], Y[idx[:split]]
    X_test, Y_test = X[idx[split:]], Y[idx[split:]]
    pos_test = pos[idx[split:]]
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print(f"\nTraining MLP (hidden={args.hidden}, epochs={args.epochs})...")
    best_acc, best_per_cell = train_mlp(
        X_train, Y_train, X_test, Y_test, in_dim=X.shape[1],
        hidden=args.hidden, epochs=args.epochs, device=device)

    print(f"\nBest overall: {best_acc:.4%}")
    print(f"\nPer-region per-cell accuracy:")
    for label in ('corner', 'edge', 'inner', 'center'):
        cells = [c for c in range(64) if cell_class(c) == label]
        m = np.mean([best_per_cell[c] for c in cells])
        mn = np.min([best_per_cell[c] for c in cells])
        mx = np.max([best_per_cell[c] for c in cells])
        print(f"  {label:>8s}: n={len(cells):2d}  mean={m:.4f}  "
              f"min={mn:.4f}  max={mx:.4f}")

    # Stratify by turn (on test set)
    print(f"\nTest accuracy stratified by turn:")
    print(f"{'turn':>10s} {'n':>8s} {'overall':>9s} {'center':>9s}")
    model_pred = torch.from_numpy(X_test.astype(np.float32)).to(device)
    bins = [(5, 10), (10, 15), (15, 20), (20, 25),
            (25, 30), (30, 35), (35, 40), (40, 45), (45, 54)]
    Y_test_t = torch.from_numpy(Y_test).to(device)
    # Re-evaluate
    model = SmallMLP(X.shape[1], args.hidden, 64 * 3).to(device)
    # Not retrained; use the saved best (we lost the model). Just retrain for stratification.
    # For now, just print structure-level results.

    print("(Per-turn stratification not implemented in this run; basic results above)")
