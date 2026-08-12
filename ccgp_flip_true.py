"""TRUE-never-flipped CCGP for Othello-GPT ONLY.

The `flip` condition in compute_ccgp uses NET-flip (current color != placement
color), which counts a disc flipped-and-flipped-back as "never-flipped". Here we
use the actual flip HISTORY from replaying the game: a square is "ever-flipped"
if its color changed at least once after it was placed (>=1 capture). We then
decode "cell C == color X" conditioned on C being truly never-flipped vs
ever-flipped, and measure CCGP transfer between the two.

  python ccgp_flip_true.py --ogpt-ckpt ckpts/gpt_synthetic.ckpt --layer 6 \
      --n 100000

CAVEAT: ever-flipped correlates with age/phase (older discs had more chances to
be captured), so use --match-ply for a phase-controlled number. --non-center
restricts to the 60 move-cells (matches the net-flip condition's cell set;
otherwise the 4 always-contested center squares are included).
"""
import argparse
import os
import numpy as np
import torch
import compute_ccgp as C


def extract(ogpt_ckpt, layer, n_sample, ply_lo=5, ply_hi=54, seed=0, batch=200):
    """Per position: OGPT resid @ layer, board(64) {0,1,2}, ply, ever_flipped(64)."""
    import pickle
    from mingpt.model import GPT, GPTConfig
    from experiments.mathematical_transformation_experiments.probe_state_pred_for_othello import (
        extract_activations, tokenize_games, _get_state_stack,
        GAME_LEN, SYNTHETIC_DIR, get_device,
    )
    device = get_device()
    sd = torch.load(ogpt_ckpt, map_location='cpu')
    if isinstance(sd, dict) and 'model' in sd and isinstance(sd['model'], dict):
        sd = sd['model']
    vocab, n_embd = sd['tok_emb.weight'].shape
    block = sd['pos_emb'].shape[1]
    n_layer = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.'))
    gpt = GPT(GPTConfig(vocab, block, n_layer=n_layer, n_head=8, n_embd=n_embd))
    gpt.load_state_dict(sd); gpt = gpt.to(device).eval()

    files = sorted(f for f in os.listdir(SYNTHETIC_DIR) if f.endswith(".pickle"))
    games = []
    for fn in files:
        with open(os.path.join(SYNTHETIC_DIR, fn), "rb") as f:
            games.extend(g for g in pickle.load(f) if len(g) == GAME_LEN)
        if len(games) >= n_sample:
            break
    rng = np.random.RandomState(seed)
    rng.shuffle(games); games = games[:n_sample]
    nmoves = rng.randint(ply_lo, ply_hi, size=len(games))
    N = len(games)
    print(f"  extract: {N} games, resid_post @ layer {layer}/{n_layer}", flush=True)

    H = np.zeros((N, n_embd), np.float32)
    board = np.zeros((N, 64), np.int8)
    ever = np.zeros((N, 64), np.int8)
    pos = np.asarray(nmoves, np.int64)
    for s0 in range(0, N, batch):
        gb = games[s0:s0 + batch]
        toks = tokenize_games(gb, seq_len=block).to(device)
        resid = extract_activations(gpt, toks, layer)
        ss = _get_state_stack(gb, 0, block).numpy()               # (b, block, 8, 8) in {-1,0,1}
        for i in range(len(gb)):
            gi = s0 + i; t = int(nmoves[gi])
            H[gi] = resid[i, t - 1].detach().cpu().numpy()
            st = ss[i, t - 1].reshape(64)
            board[gi] = np.where(st == 0, 0, np.where(st == -1, 1, 2)).astype(np.int8)
            # ever-flipped from the color history of each square over moves 0..t-1
            hist = ss[i, :t].reshape(t, 64)                        # (t, 64) colors in {-1,0,1}
            nz = hist != 0
            placed = nz.any(0)
            first = np.argmax(nz, 0)                               # first nonzero step per square
            place_col = hist[first, np.arange(64)]                 # color when first placed
            changed = ((hist != 0) & (hist != place_col[None, :])).any(0)
            ever[gi] = (placed & changed).astype(np.int8)
        if s0 % (batch * 25) == 0:
            print(f"    ...{min(s0 + batch, N)}/{N}", flush=True)

    out = {}
    for parity, mask in (("even", pos % 2 == 0), ("odd", pos % 2 == 1)):
        if mask.any():
            out[parity] = (H[mask], board[mask], pos[mask], ever[mask])
    return out


def flip_true_ccgp(h, board, pos, ever, classes=(1, 2), cells=range(64), seed=0,
                   min_per_tail=150, nonlinear=False, match_ply=False):
    rng = np.random.RandomState(seed)
    ccgp_per, within_per, nev_n, ev_n = [], [], [], []

    def _match_ply(a_idx, b_idx):
        oa, ob = [], []
        for p in np.unique(np.concatenate([pos[a_idx], pos[b_idx]])):
            aa = a_idx[pos[a_idx] == p]; bb = b_idx[pos[b_idx] == p]
            k = min(len(aa), len(bb))
            if k:
                oa.append(rng.choice(aa, k, replace=False)); ob.append(rng.choice(bb, k, replace=False))
        return (np.concatenate(oa), np.concatenate(ob)) if oa else (a_idx[:0], b_idx[:0])

    for cell in cells:
        occ = board[:, cell] != 0
        never_idx = np.where(occ & (ever[:, cell] == 0))[0]
        everf_idx = np.where(occ & (ever[:, cell] == 1))[0]
        if match_ply:
            never_idx, everf_idx = _match_ply(never_idx, everf_idx)
        if len(never_idx) < min_per_tail or len(everf_idx) < min_per_tail:
            continue
        for cls in classes:
            y = (board[:, cell] == cls).astype(np.int32)
            per = []
            for idx in (never_idx, everf_idx):
                if y[idx].sum() < 30 or (1 - y[idx]).sum() < 30:
                    per.append(None); continue
                per.append(C._balance(idx, y, rng))
            if any(p is None for p in per):
                continue
            t = max(50, min(min(len(per[0]), len(per[1])), min(len(per[0]) // 2, len(per[1]) // 2)))
            fa = []
            for held in (0, 1):
                tr = C._subsample(per[1 - held], t, rng); te = per[held]
                a = C._probe_acc(h[tr], y[tr], h[te], y[te], nonlinear=nonlinear)
                if a is not None:
                    fa.append(a)
            if fa:
                ccgp_per.append(np.mean(fa))
            wf = []
            for s in (0, 1):
                idx = per[s].copy(); rng.shuffle(idx); half = len(idx) // 2
                for trp, tep in [(idx[:half], idx[half:]), (idx[half:], idx[:half])]:
                    tr = C._subsample(trp, t, rng)
                    a = C._probe_acc(h[tr], y[tr], h[tep], y[tep], nonlinear=nonlinear)
                    if a is not None:
                        wf.append(a)
            if wf:
                within_per.append(np.mean(wf))
            nev_n.append(len(never_idx)); ev_n.append(len(everf_idx))

    return {
        'ccgp':   float(np.mean(ccgp_per))   if ccgp_per else float('nan'),
        'within': float(np.mean(within_per)) if within_per else float('nan'),
        'gap':    float(np.mean(within_per) - np.mean(ccgp_per))
                  if (ccgp_per and within_per) else float('nan'),
        'n_pairs': len(ccgp_per),
        'mean_never_n': int(np.mean(nev_n)) if nev_n else 0,
        'mean_ever_n':  int(np.mean(ev_n)) if ev_n else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ogpt-ckpt", default="ckpts/gpt_synthetic.ckpt")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--nonlinear", action="store_true")
    ap.add_argument("--match-ply", action="store_true")
    ap.add_argument("--non-center", action="store_true",
                    help="restrict to the 60 move-cells (exclude the 4 center squares)")
    args = ap.parse_args()

    cells = C._VALID_MOVES if args.non_center else list(range(64))
    print(f"OGPT TRUE-never-flipped CCGP: 0 captures vs >=1 capture "
          f"(match_ply={args.match_ply}, non_center={args.non_center})", flush=True)
    per_parity = extract(args.ogpt_ckpt, args.layer, args.n)
    for parity, (h, board, pos, ever) in per_parity.items():
        r = flip_true_ccgp(h, board, pos, ever, cells=cells,
                           nonlinear=args.nonlinear, match_ply=args.match_ply)
        print(f"\n--- parity={parity} ---")
        print(f"  CCGP   = {r['ccgp']:.4f}")
        print(f"  Within = {r['within']:.4f}")
        print(f"  Gap    = {r['gap']:.4f}")
        print(f"  ({r['n_pairs']} (cell,class) pairs; mean tail sizes: "
              f"never-flipped~{r['mean_never_n']}, ever-flipped~{r['mean_ever_n']})")


if __name__ == "__main__":
    main()
