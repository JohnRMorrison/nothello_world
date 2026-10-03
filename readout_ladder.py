"""Readout ladder: what computation does legality need, given a perfect board?

The true-board control showed the board-state trunk is NOT the bottleneck -- a
LINEAR readout of the ground-truth board leaves 26% illegal mass, about what the
trunk gives (24.25%).  So the question is which readout recovers legality, and
it can be answered with no trunk in the loop at all.

  rung 2  MLP on the board            -- is it capacity?
  rung 3  self-attention over the 64 squares -- is it relational structure?
  rung 4  960 flanking patterns + linear     -- is it the right features?

Target is uniform-over-legal, not the move actually played.  The synthetic games
ARE random.choice(legal), so uniform-over-legal is the Bayes-optimal next-move
distribution; using it directly is the same optimum with less noise, and matches
train_next_rule.py's soft-target convention.  The loss floor is E[ln n_legal]
either way.

    python readout_ladder.py --train-dir /workspace/split/train \
        --test-dir /workspace/split/test --train-pos 2000000
"""
import argparse, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from midgame_tree_mlp import sample_midgame_positions
from flanking_patterns import load_patterns, compute_pattern_activations

CENTER = {27, 28, 35, 36}
VALID = [c for c in range(64) if c not in CENTER]


def collect(pickle_dir, n_pos, seed, patterns, ply_min=5, ply_max=54):
    """-> board one-hot (N,192) f32, patterns (N,960) f16, legal (N,60) bool."""
    t0 = time.time()
    n_games = int(n_pos / (ply_max - ply_min) * 1.15) + 50
    X, S, T, L = sample_midgame_positions(
        n_games, ply_min=ply_min, ply_max=ply_max, seed=seed,
        collect_legal_moves=True, canonicalize_mover=True, pickle_dir=pickle_dir)
    X = np.asarray(X); S = np.asarray(S); L = np.asarray(L)
    keep = L[:, VALID].sum(1) > 0            # skip positions with no legal move
    X, S, L = X[keep][:n_pos], S[keep][:n_pos], L[keep][:n_pos]
    B = np.zeros((len(S), 192), np.float32)
    idx = np.arange(64) * 3
    for k in range(3):
        B[:, idx + k] = (S == k)
    P = compute_pattern_activations(patterns, X[:, :60].astype(np.uint8),
                                    X[:, 60:120].astype(np.uint8),
                                    X[:, 120].astype(np.uint8))
    P = np.asarray(P, dtype=np.float16)
    Lg = L[:, VALID].astype(bool)
    print(f'  {pickle_dir}: {len(B):,} positions, patterns {P.shape} '
          f'({time.time()-t0:.0f}s)', flush=True)
    return B, P, Lg


class SquareAttn(nn.Module):
    """Rung 3: the 64 squares as tokens, so a square can gather what lies along
    its rays.  Row and column embeddings give it coordinates to do that with."""

    def __init__(self, d=128, layers=2, heads=4):
        super().__init__()
        self.state = nn.Embedding(3, d)
        self.row = nn.Embedding(8, d)
        self.col = nn.Embedding(8, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d, 1)
        r = torch.arange(64) // 8; c = torch.arange(64) % 8
        self.register_buffer('rid', r); self.register_buffer('cid', c)
        self.register_buffer('valid', torch.tensor(VALID))

    def forward(self, b192):
        s = b192.view(-1, 64, 3).argmax(-1)                  # (B, 64) in {0,1,2}
        h = self.state(s) + self.row(self.rid) + self.col(self.cid)
        h = self.tr(h)
        return self.out(h).squeeze(-1)[:, self.valid]         # (B, 60)


def soft_ce(logits, legal):
    q = legal.float(); q = q / q.sum(1, keepdim=True)
    return -(q * F.log_softmax(logits, 1)).sum(1).mean()


@torch.no_grad()
def score(logits, legal, ks=(1, 3, 5)):
    lg = logits.float(); L = legal
    nl = L.sum(1)
    order = torch.argsort(lg, 1, descending=True)
    out = {}
    for k in ks:
        ke = torch.clamp(nl, max=k)
        gl = torch.gather(L, 1, order)                       # legality in rank order
        cum = gl.float().cumsum(1)
        h = cum.gather(1, (ke - 1).unsqueeze(1)).squeeze(1)
        out[f'top{k}'] = float((h / ke.float()).mean())
    p = torch.softmax(lg, 1)
    ill = (p * ~L).sum(1)
    out['med_ill'] = float(ill.median()); out['mean_ill'] = float(ill.mean())
    out['loss'] = float(soft_ce(lg, L))
    return out


def run(name, model, Ftr, Ltr, Fte, Lte, dev, epochs, bs=4096, lr=3e-3):
    model = model.to(dev)
    n = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    spe = (len(Ftr) + bs - 1) // bs
    total = epochs * spe
    # OneCycleLR divides by zero when a phase rounds to zero length, which it
    # does for small `total` -- so only schedule when there are enough steps.
    sch = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total,
                                               pct_start=0.1) if total >= 50 else
           torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=1))
    t0 = time.time()
    for ep in range(epochs):
        perm = np.random.permutation(len(Ftr))
        for i in range(0, len(Ftr), bs):
            j = perm[i:i + bs]
            f = torch.from_numpy(Ftr[j]).float().to(dev)
            l = torch.from_numpy(Ltr[j]).to(dev)
            loss = soft_ce(model(f), l)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    model.eval()
    acc = []
    with torch.no_grad():
        for i in range(0, len(Fte), bs):
            f = torch.from_numpy(Fte[i:i+bs]).float().to(dev)
            acc.append(model(f).cpu())
    r = score(torch.cat(acc), torch.from_numpy(Lte))
    print(f'\n{name}  ({n:,} params, {time.time()-t0:.0f}s)')
    print(f'  loss {r["loss"]:.4f}   top-1 {100*r["top1"]:6.2f}%  '
          f'top-3 {100*r["top3"]:6.2f}%  top-5 {100*r["top5"]:6.2f}%')
    print(f'  illegal mass: median {100*r["med_ill"]:6.3f}%  '
          f'mean {100*r["mean_ill"]:6.3f}%', flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-dir', required=True)
    ap.add_argument('--test-dir', required=True)
    ap.add_argument('--train-pos', type=int, default=2_000_000)
    ap.add_argument('--test-pos', type=int, default=200_000)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--patterns', default='hand_crafted_flanking_patterns.pt')
    ap.add_argument('--rungs', default='234')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f'device {dev}', flush=True)

    patterns = load_patterns(a.patterns)
    print(f'{len(patterns)} flanking patterns', flush=True)
    Btr, Ptr, Ltr = collect(a.train_dir, a.train_pos, 1, patterns)
    Bte, Pte, Lte = collect(a.test_dir, a.test_pos, 2, patterns)
    floor = float(np.log(Lte.sum(1)).mean())
    print(f'\nBAYES FLOOR E[ln n_legal] = {floor:.4f} nats  '
          f'(mean {Lte.sum(1).mean():.2f} legal)')
    print('reference -- linear on true board: top-1 98.98%, median illegal 26.00%')
    print('reference -- linear on frozen trunk: top-1 97.78%, median illegal 24.25%')

    if '2' in a.rungs:
        print('\n===== RUNG 2: capacity (MLP on the board) =====', flush=True)
        run('MLP 192->2048->60', nn.Sequential(nn.Linear(192,2048), nn.ReLU(),
            nn.Linear(2048,60)), Btr, Ltr, Bte, Lte, dev, a.epochs)
        run('MLP 192->2048->2048->60', nn.Sequential(nn.Linear(192,2048), nn.ReLU(),
            nn.Linear(2048,2048), nn.ReLU(), nn.Linear(2048,60)),
            Btr, Ltr, Bte, Lte, dev, a.epochs)
    if '3' in a.rungs:
        print('\n===== RUNG 3: relational (attention over 64 squares) =====', flush=True)
        run('SquareAttn d128 x2', SquareAttn(128, 2, 4), Btr, Ltr, Bte, Lte,
            dev, a.epochs, lr=1e-3)
    if '4' in a.rungs:
        print('\n===== RUNG 4: features (960 flanking patterns) =====', flush=True)
        run('Linear 960->60', nn.Linear(960, 60), Ptr, Ltr, Pte, Lte, dev, a.epochs)
    print(f'\n(floor {floor:.4f} nats; a perfect readout is top-1 100%, illegal 0%)')


if __name__ == '__main__':
    main()
