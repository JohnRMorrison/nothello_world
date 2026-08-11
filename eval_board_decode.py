"""Board-state decoding from each MLP's hidden layer (Nanda-style).

For every --mlp checkpoint (moveset/movegrid x H512/H4096):
  * LINEAR probe:     the saved probe_direct_* (Linear(hidden->64x3), even/odd) -
                      auto-derived from the mlp filename.
  * NON-LINEAR probe: NonLinearProbe (hidden->512->64x3), TRAINED here on a
                      train chunk (parity-split, matching the linear probe).

Both are decoded on a held-out chunk (ply range), and we report, for EACH probe:
  - overall per-cell board accuracy,
  - the 8x8 per-SQUARE accuracy grid (errors by square),
  - accuracy by MOVE number (per ply).

Hidden = relu(mlp.net[0](feat)); board target Y and features come from
_load_features (same frame the probes were trained in). Loads each chunk once.

Usage (pod):
  /usr/bin/python3.13 eval_board_decode.py \
    --mlps <CKDIR>/pattern_simple_direct_H{512,4096}_{playedeven,move_grid}.pt \
    --eval-chunk /workspace/feature_chunks/chunk_ext_0004.npz \
    --train-chunk /workspace/feature_chunks/chunk_ext_0003.npz \
    --ply-min 5 --ply-max 54
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import _load_features
from train_pattern_simple import DirectMLP, to_move_grid_input
from probe_multi_seed_hidden import NonLinearProbe

torch.set_num_threads(os.cpu_count() or 1)
N_CELLS, N_CLASSES, N_MOVES = 64, 3, 60
FEAT_PLAYEDEVEN = list(range(0, 60)) + list(range(120, 180))


def to_feat(Xraw, rep):
    if rep == 'move_grid':  return to_move_grid_input(Xraw)
    if rep == 'playedeven': return Xraw[:, FEAT_PLAYEDEVEN]
    raise ValueError(rep)


def load_slice(path, ply_min, ply_max, cap, seed):
    X, Y, pos = _load_features(path)
    Y = Y.numpy().astype(np.int64); pos = pos.numpy().astype(np.int64)
    keep = np.where((pos >= ply_min) & (pos < ply_max))[0]
    if len(keep) > cap:
        keep = np.sort(np.random.RandomState(seed).choice(keep, cap, replace=False))
    return X[keep], Y[keep], pos[keep]


@torch.no_grad()
def extract_hidden(Xraw, pos, rep, me, mo, device, batch=4096):
    H = me.net[0].out_features
    out = np.zeros((len(pos), H), dtype=np.float16)
    for i in range(0, len(pos), batch):
        x = to_feat(Xraw[i:i + batch].to(device), rep)
        pb = pos[i:i + batch]
        em = torch.from_numpy(pb % 2 == 0).to(device)
        h = torch.empty(len(x), H, device=device)
        if em.any():    h[em] = torch.relu(me.net[0](x[em]))
        if (~em).any(): h[~em] = torch.relu(mo.net[0](x[~em]))
        out[i:i + batch] = h.cpu().numpy().astype(np.float16)
    return out


def train_nonlinear(hid, Y, pos, hidden_dim, device, epochs=8, lr=1e-3, batch=4096):
    """Parity-split NonLinearProbe on (hidden, board target).

    Uses a MANUAL Adam loop rather than torch.optim.* -- constructing a torch
    optimizer lazily does `import torch._dynamo`, which is broken on this
    torch 2.13 / py3.13 build (the _register_fake/inspect _Dummy bug)."""
    probes = {}
    ce = nn.CrossEntropyLoss()
    b1, b2, eps = 0.9, 0.999, 1e-8
    for par, sel in (('even', pos % 2 == 0), ('odd', pos % 2 == 1)):
        h = torch.from_numpy(hid[sel]).float(); y = torch.from_numpy(Y[sel]).long()
        p = NonLinearProbe(hidden_dim).to(device)
        params = list(p.parameters())
        m = [torch.zeros_like(q) for q in params]
        v = [torch.zeros_like(q) for q in params]
        t = 0
        n = len(h); g = torch.Generator().manual_seed(0)
        for _ in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                logits = p(h[idx].to(device))                 # (b,64,3)
                loss = ce(logits.reshape(-1, N_CLASSES), y[idx].reshape(-1).to(device))
                for q in params:
                    q.grad = None
                loss.backward()
                t += 1
                with torch.no_grad():
                    for j, q in enumerate(params):
                        gr = q.grad
                        m[j].mul_(b1).add_(gr, alpha=1 - b1)
                        v[j].mul_(b2).addcmul_(gr, gr, value=1 - b2)
                        mhat = m[j] / (1 - b1 ** t)
                        vhat = v[j] / (1 - b2 ** t)
                        q.addcdiv_(mhat, vhat.sqrt().add_(eps), value=-lr)
        p.eval(); probes[par] = p
    return probes['even'], probes['odd']


@torch.no_grad()
def decode_metrics(hid, Y, pos, probe_even, probe_odd, is_linear, device, batch=8192):
    """preds from a probe; accumulate per-cell and per-ply hits."""
    cell_correct = np.zeros(64, np.int64); cell_total = np.zeros(64, np.int64)
    ply_correct = np.zeros(60, np.int64); ply_total = np.zeros(60, np.int64)
    for i in range(0, len(pos), batch):
        h = torch.from_numpy(hid[i:i + batch]).float().to(device)
        pb = pos[i:i + batch]; yb = Y[i:i + batch]
        preds = np.zeros((len(pb), 64), np.int64)
        for par, sel in (('e', pb % 2 == 0), ('o', pb % 2 == 1)):
            if not sel.any(): continue
            pr = probe_even if par == 'e' else probe_odd
            hs = h[torch.from_numpy(sel).to(device)]
            lg = (pr(hs).view(-1, 64, 3) if is_linear else pr(hs))   # linear: view; nlp: already (b,64,3)
            preds[sel] = lg.argmax(-1).cpu().numpy()
        hit = (preds == yb)                                          # (b,64)
        cell_correct += hit.sum(0); cell_total += len(pb)
        np.add.at(ply_correct, pb, hit.sum(1))
        np.add.at(ply_total, pb, 64)
    return cell_correct, cell_total, ply_correct, ply_total


def report(name, cell_c, cell_t, ply_c, ply_t):
    acc = cell_c / np.maximum(cell_t, 1)
    print(f"\n----- {name} -----")
    print(f"overall per-cell acc (all 64 squares): {acc.mean():.4f}")
    print("8x8 per-square accuracy (rank 1 top):")
    g = acc.reshape(8, 8)
    print("      " + "  ".join('ABCDEFGH'))
    for r in range(8):
        print(f"  {r+1}  " + "  ".join(f"{g[r,c]*100:4.1f}" for c in range(8)))
    print("accuracy by move number (ply):")
    for p in range(60):
        if ply_t[p]:
            print(f"    ply {p:2d}: {100*ply_c[p]/ply_t[p]:.2f}%  (n={ply_t[p]//64})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mlps', nargs='+', required=True)
    ap.add_argument('--eval-chunk', required=True)
    ap.add_argument('--train-chunk', default=None,
                    help="Only needed WITHOUT --load-nonlinear (to train the floor probe).")
    ap.add_argument('--ply-min', type=int, default=5)
    ap.add_argument('--ply-max', type=int, default=54)
    ap.add_argument('--eval-positions', type=int, default=300_000)
    ap.add_argument('--train-positions', type=int, default=200_000)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--load-nonlinear', action='store_true',
                    help="Load a pretrained non-linear probe (probe_nonlinear_direct_*, "
                         "trained on the full chunk set by probe_pattern_models.py "
                         "--nonlinear) instead of training one here. This is the FAIR "
                         "comparison to the 6M-game linear probe.")
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device} threads={torch.get_num_threads()}', flush=True)

    print('loading EVAL chunk ...', flush=True)
    Xe, Ye, pose = load_slice(args.eval_chunk, args.ply_min, args.ply_max, args.eval_positions, 0)
    Xt = Yt = post = None
    if not args.load_nonlinear:
        if args.train_chunk is None:
            ap.error("--train-chunk is required unless --load-nonlinear is set")
        print('loading TRAIN chunk ...', flush=True)
        Xt, Yt, post = load_slice(args.train_chunk, args.ply_min, args.ply_max, args.train_positions, 1)
    print(f'eval N={len(pose):,}' + (f'  train N={len(post):,}' if post is not None else ''), flush=True)

    for mlp_path in args.mlps:
        ck = torch.load(mlp_path, map_location='cpu', weights_only=False)
        input_dim, H = ck['input_dim'], ck['hidden_dim']
        rep = {120: 'playedeven', 3600: 'move_grid'}[input_dim]
        me = DirectMLP(input_dim, H).to(device); me.load_state_dict(ck['even']); me.eval()
        mo = DirectMLP(input_dim, H).to(device); mo.load_state_dict(ck['odd']); mo.eval()
        probe_path = mlp_path.replace('pattern_simple_direct_', 'probe_direct_')
        pk = torch.load(probe_path, map_location='cpu', weights_only=False)
        lin_e = nn.Linear(H, 192).to(device); lin_e.load_state_dict(pk['even']); lin_e.eval()
        lin_o = nn.Linear(H, 192).to(device); lin_o.load_state_dict(pk['odd']); lin_o.eval()
        print(f"\n######### {os.path.basename(mlp_path)}  (rep={rep}, H={H}) #########", flush=True)

        # non-linear probe: load a full-chunk-trained one (fair) or train here (floor)
        if args.load_nonlinear:
            nl_path = mlp_path.replace('pattern_simple_', 'probe_nonlinear_')
            nk = torch.load(nl_path, map_location='cpu', weights_only=False)
            nlp_e = NonLinearProbe(H).to(device); nlp_e.load_state_dict(nk['even']); nlp_e.eval()
            nlp_o = NonLinearProbe(H).to(device); nlp_o.load_state_dict(nk['odd']); nlp_o.eval()
            nl_tag = f"NON-LINEAR probe (loaded {os.path.basename(nl_path)}, full-chunk)"
            print(f"  loaded non-linear probe: {nl_path}  (best_acc={nk.get('best_acc')})", flush=True)
        else:
            hid_t = extract_hidden(Xt, post, rep, me, mo, device)
            nlp_e, nlp_o = train_nonlinear(hid_t, Yt, post, H, device, epochs=args.epochs)
            nl_tag = f"NON-LINEAR probe (trained here, {args.train_positions//1000}k pos -- FLOOR)"
        hid_e = extract_hidden(Xe, pose, rep, me, mo, device)

        lc, lt, lpc, lpt = decode_metrics(hid_e, Ye, pose, lin_e, lin_o, True, device)
        report(f"{os.path.basename(mlp_path)}  LINEAR probe (saved probe_direct)", lc, lt, lpc, lpt)
        nc, nt, npc, npt = decode_metrics(hid_e, Ye, pose, nlp_e, nlp_o, False, device)
        report(f"{os.path.basename(mlp_path)}  {nl_tag}", nc, nt, npc, npt)


if __name__ == '__main__':
    main()
