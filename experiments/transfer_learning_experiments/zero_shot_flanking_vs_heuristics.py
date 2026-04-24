"""
Zero-shot companion to the flanking-vs-heuristics transfer-learning experiment.

Compares three per-cell legal-move predictors on a shared set of standard
Othello positions:

    1. FLANKING       — the 960 hand-crafted flanking patterns
                        (HandCraftedFlanking in hand_crafted_flanking.py).
                        No training; analytic weights.
    2. HEURISTIC      — the bag-of-heuristics predictor from
                        reverse_engineering_experiments/heuristic_legal_move_predictor.py:
                        for each of the ~902 neurons, a boolean "any rule fires"
                        feature; a linear readout (trained on an 80% split)
                        maps those features to per-cell legality.
    3. PRETRAINED_GPT — the pretrained OthelloGPT. Its top-1 next-move
                        prediction is checked for legality; per-cell legality
                        is derived from softmax ≥ threshold.

Outputs JSON with:
    - per-predictor accuracy / precision / recall / F1 against ground truth
    - agreement matrix: P(predictor_a == predictor_b | per cell)
    - error-explanation rates: on positions where the pretrained model's top-1
      is illegal, what fraction of those errors does flanking vs. heuristics
      "explain" (i.e., also mark that cell as legal)

Usage:
    python zero_shot_flanking_vs_heuristics.py \\
        --rules ../reverse_engineering_experiments/rules_085_200_2-6.json \\
        --ckpt ../../ckpts/gpt_synthetic.ckpt \\
        --n-games 1000 \\
        --output zero_shot_flanking_vs_heuristics.json

    # Tiny smoke test (< 1 min):
    python zero_shot_flanking_vs_heuristics.py \\
        --rules ../reverse_engineering_experiments/rules_085_200_2-6.json \\
        --n-games 50 --epochs 5 --output /tmp/z.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_REV_ENG = os.path.join(_REPO_ROOT, "experiments", "reverse_engineering_experiments")
for _p in (_HERE, _REPO_ROOT, _REV_ENG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.othello import OthelloBoardState, get_ood_game  # noqa: E402
from mingpt.model import GPT, GPTConfig  # noqa: E402
from mingpt.dataset import CharDataset  # noqa: E402

from hand_crafted_flanking import (  # noqa: E402
    HandCraftedFlanking, VALID_MOVES, MOVE_TO_IDX, N_MOVES,
    encode_board, enumerate_flanking_patterns,
)

# Heuristic predictor from the reverse_engineering_experiments suite.
# Its module imports OthelloReverseEngineering.utils.*; we just need the
# three helper functions.
from heuristic_legal_move_predictor import (  # noqa: E402
    load_and_parse_rules, evaluate_heuristics, train_linear_model,
)
from OthelloReverseEngineering.utils.othello_utils import (  # noqa: E402
    games_batch_to_board_state_flipped_played_BLC,
    games_batch_to_valid_moves_BLRRC,
)
from OthelloReverseEngineering.utils.circuits_utils import (  # noqa: E402
    construct_othello_dataset,
)


# ---------------------------------------------------------------------------
# Feature-channel projection (960 → 60 playable cells → 64 board squares)
# ---------------------------------------------------------------------------

def flanking60_to_board64(legal60):
    """Expand (N, 60) flanking-style outputs to (N, 64), filling the 4 center
    cells with 0 (they're never playable so never legal)."""
    N = legal60.shape[0]
    out = np.zeros((N, 64), dtype=np.float32)
    for i, pos in enumerate(VALID_MOVES):
        out[:, pos] = legal60[:, i]
    return out


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pretrained_gpt(ckpt_path, device):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mconf = GPTConfig(61, 59, n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def build_token_maps():
    dummy_games = [get_ood_game(i) for i in range(10)]
    ds = CharDataset(dummy_games)
    return ds.stoi, ds.itos


# ---------------------------------------------------------------------------
# Pretrained-GPT per-cell legality (per position, from softmax)
# ---------------------------------------------------------------------------

def gpt_per_cell_legality(model, games, stoi, itos, device,
                           threshold=0.02, max_seq_len=59):
    """Run the pretrained model over `games` and return:

      - argmax_moves: (N, ) board-positions of GPT's top-1 next move
      - per_cell:    (N, 64) bool — GPT's "legal" prediction for each square
                     (True if softmax on that square's token ≥ threshold)

    N is total (game, position) pairs across all games.
    """
    argmax_moves = []
    per_cell = []

    with torch.no_grad():
        for game in games:
            if len(game) < 2:
                continue
            encoded = [stoi[m] for m in game]
            if len(encoded) > max_seq_len + 1:
                encoded = encoded[: max_seq_len + 1]
            x = torch.tensor(encoded[:-1], dtype=torch.long)[None].to(device)
            logits, _ = model(x)
            probs = F.softmax(logits[0], dim=-1).cpu().numpy()  # (L-1, vocab)

            for pos in range(probs.shape[0]):
                # Top-1 move
                pred_token = int(probs[pos].argmax())
                pred_move = int(itos[pred_token])
                argmax_moves.append(pred_move)
                # Per-cell legality
                row = np.zeros(64, dtype=bool)
                for tok_idx, square_pos in itos.items():
                    if square_pos == -100:
                        continue
                    if 0 <= square_pos < 64 and probs[pos, tok_idx] >= threshold:
                        row[square_pos] = True
                per_cell.append(row)

    return np.array(argmax_moves), np.array(per_cell)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _binary_metrics(pred, gt):
    """pred, gt: boolean arrays of same shape; returns dict of metrics over
    the flattened arrays."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    tn = int(np.sum(~pred & ~gt))
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _agreement(a, b):
    """P(a == b) across matched boolean entries."""
    a = a.astype(bool).ravel()
    b = b.astype(bool).ravel()
    return round(float(np.mean(a == b)), 6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot comparison of flanking and bag-of-heuristics "
                    "legal-move predictors against the pretrained OthelloGPT.")
    parser.add_argument("--rules", type=str, required=True,
                        help="Rules JSON from extract_rules.py "
                             "(same file used by run_2x2.sh).")
    parser.add_argument("--ckpt", type=str, default="../../ckpts/gpt_synthetic.ckpt",
                        help="Pretrained OthelloGPT checkpoint.")
    parser.add_argument("--n-games", type=int, default=1000,
                        help="Games used to build the test set (80/20 split).")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Epochs for training the heuristic linear readout.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpt-threshold", type=float, default=0.02,
                        help="Softmax threshold for GPT per-cell legal prediction.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path. If omitted, prints summary only.")
    parser.add_argument("--skip-gpt", action="store_true",
                        help="Skip pretrained-GPT evaluation (cheaper smoke run).")
    args = parser.parse_args()

    # --- Device ---
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}", flush=True)

    # --- Load dataset via the OthelloReverseEngineering helpers (so we get
    #     the exact 320-dim feature representation the heuristic rules were
    #     extracted against) ---
    print(f"\nLoading {args.n_games} games + features (320-dim)...", flush=True)
    dataset = construct_othello_dataset(
        custom_functions=[], n_inputs=args.n_games,
        split="train", precompute_dataset=False,
    )
    decoded_inputs = dataset["decoded_inputs"]

    features_BLC = games_batch_to_board_state_flipped_played_BLC(decoded_inputs)
    valid_moves_BLRRC = games_batch_to_valid_moves_BLRRC(decoded_inputs)
    B, L = features_BLC.shape[:2]
    valid_moves_BL64 = valid_moves_BLRRC.reshape(B, L, 64)

    # GPT emits L-1 predictions per game (one per "given moves 0..t, predict t+1"),
    # so to keep everything index-aligned across predictors we drop the last turn
    # of every game up front. Otherwise indices offset by +1 per game boundary
    # when we try to match GPT predictions against features/labels.
    features_trim = features_BLC[:, :L - 1, :].reshape(-1, 320).float()
    labels_trim = valid_moves_BL64[:, :L - 1, :].reshape(-1, 64).float()
    features_flat = features_trim
    labels_flat = labels_trim
    N = features_flat.shape[0]
    L_eff = L - 1
    print(f"  {N} positions ({B} games x {L_eff} turns); positive rate "
          f"{labels_flat.mean().item()*100:.1f}%", flush=True)

    # 80/20 split (seeded)
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(N, generator=g)
    split = int(0.8 * N)
    train_idx, test_idx = perm[:split], perm[split:]

    y_train = labels_flat[train_idx]
    y_test = labels_flat[test_idx]
    gt_test = y_test.numpy().astype(bool)
    print(f"  Train: {len(train_idx)} positions, Test: {len(test_idx)}",
          flush=True)

    # --- Heuristic predictor (train on 80%, eval on 20%) ---
    print("\n[1] Heuristic: loading rules + evaluating per position...",
          flush=True)
    parsed_rules, neuron_keys = load_and_parse_rules(args.rules)
    heuristic_all = evaluate_heuristics(features_flat, parsed_rules, neuron_keys)
    X_train_heur = heuristic_all[train_idx]
    X_test_heur = heuristic_all[test_idx]
    print(f"  Heuristic feature dim: {X_train_heur.shape[1]}  "
          f"(active fraction {heuristic_all.mean().item()*100:.1f}%)",
          flush=True)

    print("  Training linear readout...", flush=True)
    # train_linear_model returns dict of metrics; we want the raw preds on test,
    # so re-implement the eval with the trained model below.
    import torch.nn as nn  # local
    from torch.utils.data import DataLoader, TensorDataset

    readout = nn.Linear(X_train_heur.shape[1], 64).to(device)
    optim = torch.optim.Adam(readout.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(X_train_heur, y_train),
        batch_size=args.batch_size, shuffle=True,
    )
    for epoch in range(args.epochs):
        readout.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(readout(xb), yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    epoch {epoch+1}/{args.epochs}  loss={loss.item():.4f}",
                  flush=True)

    readout.eval()
    with torch.no_grad():
        heur_preds_list = []
        for xb, _ in DataLoader(
            TensorDataset(X_test_heur, y_test), batch_size=args.batch_size,
        ):
            heur_preds_list.append(
                (torch.sigmoid(readout(xb.to(device))) > 0.5).cpu().numpy()
            )
    heur_pred_test = np.concatenate(heur_preds_list, axis=0)
    print(f"  Heuristic test prediction shape: {heur_pred_test.shape}",
          flush=True)

    # --- Flanking predictor (analytic, 960 patterns) ---
    print("\n[2] Flanking: running 960-pattern network on TEST positions...",
          flush=True)
    patterns = enumerate_flanking_patterns()
    # Build one network (mine / opponent encoding comes from encode_board with
    # is_even_turn flag — here we regenerate each snapshot via the Othello
    # engine so we know whose turn it is).
    net_even = HandCraftedFlanking(patterns).eval().to(device)
    net_odd = HandCraftedFlanking(patterns).eval().to(device)

    print("  Rebuilding board snapshots from decoded_inputs...", flush=True)
    # Iterate L_eff (= L - 1) turns per game so the flattened order matches
    # features_flat / labels_flat (which we trimmed to L-1 above).
    flanking_full = np.zeros((N, 64), dtype=np.float32)
    pos_idx = 0
    t0 = time.time()
    for gi, game in enumerate(decoded_inputs):
        board = OthelloBoardState()
        for turn_idx in range(L_eff):
            if turn_idx < len(game):
                board.umpire(int(game[turn_idx]))
            is_black_turn = (board.next_hand_color == 1)
            x = encode_board(board.state, is_black_turn).unsqueeze(0).to(device)
            with torch.no_grad():
                net = net_even if is_black_turn else net_odd
                pred = net(x).squeeze(0).cpu().numpy()
            row64 = np.zeros(64, dtype=np.float32)
            for i, pos in enumerate(VALID_MOVES):
                row64[pos] = 1.0 if pred[i] > 0.5 else 0.0
            flanking_full[pos_idx] = row64
            pos_idx += 1
        if (gi + 1) % max(1, len(decoded_inputs) // 10) == 0:
            el = time.time() - t0
            print(f"    {gi+1}/{len(decoded_inputs)} games  "
                  f"(elapsed {el:.1f}s)", flush=True)

    flanking_pred_test = flanking_full[test_idx.numpy()].astype(bool)
    print(f"  Flanking test prediction shape: {flanking_pred_test.shape}",
          flush=True)

    # --- Pretrained GPT (optional: slower, per-game scan) ---
    gpt_argmax_test = None
    gpt_percell_test = None
    if not args.skip_gpt:
        print("\n[3] Pretrained GPT: per-cell + argmax...", flush=True)
        ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else \
            os.path.join(_HERE, args.ckpt)
        gpt = load_pretrained_gpt(ckpt_path, device)
        stoi, itos = build_token_maps()
        # Run GPT on the same games, L-1 predictions per game — matches
        # the L_eff trimming of features_flat / labels_flat exactly.
        argmax_full, percell_full = gpt_per_cell_legality(
            gpt, decoded_inputs, stoi, itos, device,
            threshold=args.gpt_threshold, max_seq_len=L_eff,
        )
        if argmax_full.shape[0] != N:
            print(f"  WARNING: GPT produced {argmax_full.shape[0]} predictions, "
                  f"expected {N}. Clipping to min length.", flush=True)
            M = min(argmax_full.shape[0], N)
            argmax_full = argmax_full[:M]
            percell_full = percell_full[:M]
            test_np = test_idx.numpy()
            keep = test_np[test_np < M]
            # Map test-idx -> heur_pred_test row
            test_to_heur = {int(t): i for i, t in enumerate(test_np)}
            heur_keep = np.array([test_to_heur[int(k)] for k in keep])
            gpt_argmax_test = argmax_full[keep]
            gpt_percell_test = percell_full[keep]
            gt_test_gpt = labels_flat.numpy().astype(bool)[keep]
            flanking_test_gpt = flanking_full.astype(bool)[keep]
            heur_test_gpt = heur_pred_test[heur_keep]
        else:
            # Clean alignment: test_idx addresses the same positions across
            # features_flat / labels_flat / flanking_full / argmax_full.
            test_np = test_idx.numpy()
            gpt_argmax_test = argmax_full[test_np]
            gpt_percell_test = percell_full[test_np]
            gt_test_gpt = labels_flat.numpy().astype(bool)[test_np]
            flanking_test_gpt = flanking_full.astype(bool)[test_np]
            heur_test_gpt = heur_pred_test  # already ordered by test_idx
    else:
        print("\n[3] Pretrained GPT: SKIPPED (--skip-gpt)", flush=True)
        gt_test_gpt = None
        flanking_test_gpt = None
        heur_test_gpt = None

    # --- Metrics ---
    print("\n" + "=" * 60, flush=True)
    print("Per-cell accuracy (on TEST split):", flush=True)
    print("=" * 60, flush=True)
    flanking_metrics = _binary_metrics(flanking_pred_test, gt_test)
    heur_metrics_dict = _binary_metrics(heur_pred_test, gt_test)
    print(f"  Flanking     : acc={flanking_metrics['accuracy']:.4f}  "
          f"prec={flanking_metrics['precision']:.4f}  "
          f"rec={flanking_metrics['recall']:.4f}  "
          f"f1={flanking_metrics['f1']:.4f}", flush=True)
    print(f"  Heuristic    : acc={heur_metrics_dict['accuracy']:.4f}  "
          f"prec={heur_metrics_dict['precision']:.4f}  "
          f"rec={heur_metrics_dict['recall']:.4f}  "
          f"f1={heur_metrics_dict['f1']:.4f}", flush=True)
    result = {
        "meta": {
            "n_games": args.n_games,
            "n_positions_total": int(N),
            "n_test": int(labels_flat.shape[0] - split),
            "rules_file": args.rules,
            "ckpt": args.ckpt,
            "gpt_threshold": args.gpt_threshold,
            "epochs": args.epochs,
        },
        "flanking_vs_gt": flanking_metrics,
        "heuristic_vs_gt": heur_metrics_dict,
        "agreement": {
            "flanking_vs_heuristic_cells":
                _agreement(flanking_pred_test, heur_pred_test),
        },
    }

    if not args.skip_gpt:
        gpt_percell_metrics = _binary_metrics(gpt_percell_test, gt_test_gpt)
        print(f"  Pretrained GPT: acc={gpt_percell_metrics['accuracy']:.4f}  "
              f"prec={gpt_percell_metrics['precision']:.4f}  "
              f"rec={gpt_percell_metrics['recall']:.4f}  "
              f"f1={gpt_percell_metrics['f1']:.4f}", flush=True)

        # Top-1 legality (how often GPT's single best move is actually legal)
        # gt_test_gpt is (n_keep, 64) — legal set per position.
        top1_legal_mask = np.array([
            bool(gt_test_gpt[i, m]) if 0 <= m < 64 else False
            for i, m in enumerate(gpt_argmax_test)
        ])
        top1_legal_rate = float(top1_legal_mask.mean()) if len(top1_legal_mask) else 0.0
        print(f"\n  GPT top-1 legality: {top1_legal_rate:.4f} "
              f"(fraction of next-move predictions that are in the legal set)",
              flush=True)

        # Error explanation: among positions where GPT's top-1 is illegal,
        # did flanking's per-cell map the target as legal? did heuristic?
        err_positions = ~top1_legal_mask
        n_err = int(err_positions.sum())
        if n_err > 0:
            err_moves = gpt_argmax_test[err_positions]
            err_moves_clip = np.clip(err_moves, 0, 63)
            # For each error, look up that move's column in flanking/heur
            flank_err = np.array([
                bool(flanking_test_gpt[np.where(err_positions)[0][i], m])
                for i, m in enumerate(err_moves_clip)
            ])
            heur_err = np.array([
                bool(heur_test_gpt[np.where(err_positions)[0][i], m])
                for i, m in enumerate(err_moves_clip)
            ])
            flank_explain = float(flank_err.mean())
            heur_explain = float(heur_err.mean())
            print(f"  Error explanation (n_errors={n_err}):", flush=True)
            print(f"    Flanking also predicts GPT's top-1 legal: "
                  f"{flank_explain:.4f}", flush=True)
            print(f"    Heuristic also predicts GPT's top-1 legal: "
                  f"{heur_explain:.4f}", flush=True)
            error_explanation = {
                "n_errors": n_err,
                "flanking_explanation_rate": round(flank_explain, 6),
                "heuristic_explanation_rate": round(heur_explain, 6),
            }
        else:
            error_explanation = {"n_errors": 0}

        result.update({
            "gpt_percell_vs_gt": gpt_percell_metrics,
            "gpt_top1_legal_rate": round(top1_legal_rate, 6),
            "error_explanation": error_explanation,
            "agreement": {
                **result["agreement"],
                "gpt_vs_flanking_cells":
                    _agreement(gpt_percell_test, flanking_test_gpt),
                "gpt_vs_heuristic_cells":
                    _agreement(gpt_percell_test, heur_test_gpt),
            },
        })

    # --- Interpretation hints ---
    print("\n" + "=" * 60, flush=True)
    print("Interpretation:", flush=True)
    print("=" * 60, flush=True)
    print(
        "  - Flanking F1 ≫ Heuristic F1 confirms flanking is a cleaner "
        "legal-move oracle.\n"
        "  - If GPT's per-cell predictions agree more with flanking than with\n"
        "    heuristics (see agreement table), that's a zero-shot analog of the\n"
        "    flanking hypothesis: the model's internal computation tracks\n"
        "    flanking geometry more closely than the extracted rules.\n"
        "  - Error-explanation asks the same question conditioned on GPT's\n"
        "    mistakes: does flanking or heuristic 'endorse' the illegal move\n"
        "    GPT picked?",
        flush=True,
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved results -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
