"""Train an Othello-GPT-shaped transformer to output the BOARD STATE.

The head is Linear(n_embd, 64*3): at each position it classifies every one of the
64 squares as empty / opponent's / mine.  Othello-GPT has to be *probed* for a
world model; this network is handed one as its objective, so it serves as a
positive control -- if interventions and a legal-move readout fail here too,
they are not failing for want of a board representation.

Labels are MOVER-RELATIVE by default (empty / yours / mine): the frame the 960
flanking patterns are written in, so a legality readout can sit on the output
directly.  --absolute gives empty / white / black instead.  Whose turn it is
comes from OthelloBoardState.next_hand_color, not position parity, because a
forfeit moves the turn without consuming a position.

At position t the model has seen moves 0..t, so the target is the board AFTER
t+1 moves -- the board it has just been told about, not a prediction.

Labels are cached on disk as int8 memmaps and streamed.  Holding them as int64
in RAM costs 30 GB per million games (151 GB for 5M); as int8 it is 3.8 GB per
million, and memmapped it need not be resident at all.

    python train_board_state_gpt.py --n-games 5000000 --epochs 3 --n-layer 6 \
        --cache /workspace/board_cache --out ckpts/board_gpt_L6_5M.ckpt
"""
import argparse, contextlib, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState

BLOCK = 59
CENTER = {27, 28, 35, 36}
VALID = [c for c in range(64) if c not in CENTER]
CELL_TO_TOK = {c: i + 1 for i, c in enumerate(VALID)}


class GPTBoardState(GPT):
    """Same trunk as Othello-GPT; head emits 64 squares x 3 classes."""

    def __init__(self, config):
        super().__init__(config)
        self.head = nn.Linear(config.n_embd, 64 * 3, bias=False)
        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        x = self.drop(self.tok_emb(idx) + self.pos_emb[:, :t, :])
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x).view(b, t, 64, 3)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1).long(),
                                   ignore_index=-100)
        return logits, loss


_ABS = False


def _one(g):
    """(tokens int16[59], labels int8[59,64]) for one game."""
    x = np.zeros(BLOCK, np.int16)
    y = np.full((BLOCK, 64), -100, np.int8)
    bd = OthelloBoardState()
    for t, mv in enumerate(g[:BLOCK]):
        x[t] = CELL_TO_TOK[mv]
        bd.update([mv])
        st = bd.state.flatten()
        lab = np.zeros(64, np.int8)
        if _ABS:
            lab[st == -1] = 1; lab[st == 1] = 2
        else:
            mover = bd.next_hand_color          # forfeit-aware
            lab[st == -mover] = 1; lab[st == mover] = 2
        y[t] = lab
    return x, y


def _init(absolute):
    global _ABS
    _ABS = absolute


# ---- label packing ------------------------------------------------------
# Labels take exactly four values (-100 ignore, 0 empty, 1 theirs, 2 mine), so
# two bits hold one square and one byte holds four.  That shrinks the label
# cache 4x -- 3,776 bytes/game down to 944 -- which is what lets a 20M-game
# cache fit on disk at all.  -100 is stored as code 3.


def pack_labels(y):
    """int8 (..., 64) in {-100,0,1,2}  ->  uint8 (..., 16)."""
    c = np.where(y == -100, 3, y).astype(np.uint8).reshape(*y.shape[:-1], 16, 4)
    return c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)


def unpack_labels(p, dev):
    """uint8 (..., 16) -> long tensor (..., 64) on dev, code 3 back to -100.
    Unpacked on the GPU, so a quarter as much data crosses the bus."""
    t = torch.from_numpy(np.ascontiguousarray(p)).to(dev)
    sh = torch.tensor([0, 2, 4, 6], device=dev, dtype=torch.uint8)
    c = ((t.unsqueeze(-1) >> sh) & 3).reshape(*t.shape[:-1], 64).long()
    return torch.where(c == 3, torch.full_like(c, -100), c)


def build_cache(games, prefix, absolute, nproc):
    """Write X/Y memmaps, reusing them if already present and the right size.
    Y is 2-bit packed (see pack_labels), so its last axis is 16, not 64."""
    xp, yp = prefix + '_X.npy', prefix + '_Yp.npy'
    n = len(games)
    if os.path.exists(xp) and os.path.exists(yp):
        X = np.load(xp, mmap_mode='r'); Y = np.load(yp, mmap_mode='r')
        if len(X) == n and Y.shape[-1] == 16:
            print(f'  reusing cache {xp} ({n:,} games)', flush=True)
            return X, Y
    os.makedirs(os.path.dirname(xp) or '.', exist_ok=True)
    X = np.lib.format.open_memmap(xp, mode='w+', dtype=np.int16, shape=(n, BLOCK))
    Y = np.lib.format.open_memmap(yp, mode='w+', dtype=np.uint8, shape=(n, BLOCK, 16))
    t0 = time.time()
    with Pool(nproc, initializer=_init, initargs=(absolute,)) as pool:
        for i, (x, y) in enumerate(pool.imap(_one, games, chunksize=512)):
            X[i] = x; Y[i] = pack_labels(y)
            if (i + 1) % 250_000 == 0:
                print(f'  built {i+1:,}/{n:,}  ({time.time()-t0:.0f}s)', flush=True)
    X.flush(); Y.flush()
    print(f'  cache written in {time.time()-t0:.0f}s', flush=True)
    return np.load(xp, mmap_mode='r'), np.load(yp, mmap_mode='r')


def load_games(data_dir, n_games, held_out=False, n_held=3):
    files = sorted(os.listdir(data_dir))
    files = files[-n_held:] if held_out else files[:-n_held]
    out = []
    for f in files:
        with open(os.path.join(data_dir, f), 'rb') as fh:
            out.extend(g for g in pickle.load(fh) if len(g) >= 20)
        if len(out) >= n_games:
            break
    return out[:n_games]


def _autocast(dev, amp):
    """bf16 on CUDA.  The matmuls run on Tensor Cores while weights and the
    optimizer stay fp32, which measured 0.264 s/step in fp32 against a 58,596
    step run -- 4.3 h.  bf16 needs no loss scaling, unlike fp16."""
    if dev == 'cuda' and amp:
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, X, Y, dev, batch=256, amp=True):
    """Per-square accuracy over all moves, and over moves 5-53 (the range the
    Othello-GPT probe is scored on, so the two are comparable)."""
    model.eval(); tot = np.zeros(2); hit = np.zeros(2); per_move = np.zeros((BLOCK, 2))
    for i in range(0, len(X), batch):
        x = torch.from_numpy(np.asarray(X[i:i+batch])).long().to(dev)
        y = unpack_labels(Y[i:i+batch], dev)
        with _autocast(dev, amp):
            pr = model(x)[0].argmax(-1)
        m = y != -100
        ok = (pr == y) & m
        hit[0] += int(ok.sum()); tot[0] += int(m.sum())
        hit[1] += int(ok[:, 4:53].sum()); tot[1] += int(m[:, 4:53].sum())
        per_move[:, 0] += ok.sum(-1).sum(0).cpu().numpy()
        per_move[:, 1] += m.sum(-1).sum(0).cpu().numpy()
    model.train()
    return hit[0]/max(tot[0],1), hit[1]/max(tot[1],1), per_move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--cache', default='./cache/board')
    ap.add_argument('--n-games', type=int, default=1_000_000)
    ap.add_argument('--n-eval', type=int, default=3_000)
    ap.add_argument('--n-layer', type=int, default=4)
    ap.add_argument('--n-head', type=int, default=8)
    ap.add_argument('--n-embd', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--nproc', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--absolute', action='store_true')
    ap.add_argument('--fp32', action='store_true',
                    help='disable bf16 autocast (default is bf16 on CUDA)')
    ap.add_argument('--out', default='ckpts/board_gpt.ckpt')
    a = ap.parse_args()

    dev = ('cuda' if torch.cuda.is_available()
           else 'mps' if torch.backends.mps.is_available() else 'cpu')
    amp = not a.fp32
    if dev == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f'device {dev}  nproc {a.nproc}  '
          f'precision {"bf16" if (dev == "cuda" and amp) else "fp32"}', flush=True)

    t0 = time.time()
    print('loading games...', flush=True)
    tr = load_games(a.data_dir, a.n_games)
    ev = load_games(a.data_dir, a.n_eval, held_out=True)
    print(f'{len(tr):,} train / {len(ev):,} held-out games ({time.time()-t0:.0f}s)', flush=True)
    Xtr, Ytr = build_cache(tr, f'{a.cache}_tr{len(tr)}', a.absolute, a.nproc)
    Xev, Yev = build_cache(ev, f'{a.cache}_ev{len(ev)}', a.absolute, a.nproc)
    del tr, ev

    cfg = GPTConfig(61, BLOCK, n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd)
    model = GPTBoardState(cfg).to(dev)
    print(f'{sum(p.numel() for p in model.parameters()):,} parameters', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    spe = (len(Xtr) + a.batch_size - 1) // a.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=a.epochs * spe, pct_start=0.05)
    step = 0
    for ep in range(a.epochs):
        perm = np.random.permutation(len(Xtr))
        for i in range(0, len(Xtr), a.batch_size):
            idx = np.sort(perm[i:i+a.batch_size])            # sorted = faster memmap reads
            x = torch.from_numpy(np.asarray(Xtr[idx])).long().to(dev)
            y = unpack_labels(Ytr[idx], dev)
            with _autocast(dev, amp):
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 500 == 0:
                print(f'  ep{ep} step {step}/{a.epochs*spe}  loss {loss.item():.4f}  '
                      f'({time.time()-t0:.0f}s)', flush=True)
        acc_all, acc_rng, _ = evaluate(model, Xev, Yev, dev, amp=amp)
        print(f'epoch {ep}: all moves {100*acc_all:.2f}%   moves 5-53 {100*acc_rng:.2f}%',
              flush=True)
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        torch.save({'model': model.state_dict(), 'cfg': vars(cfg), 'args': vars(a),
                    'epoch': ep}, a.out)

    acc_all, acc_rng, pm = evaluate(model, Xev, Yev, dev, amp=amp)
    print(f'\nFINAL  all moves {100*acc_all:.2f}%   moves 5-53 {100*acc_rng:.2f}%')
    print('  (Othello-GPT probe, same data: 95.84% at depth 4, 99.19% peak)')
    print('by move:', '  '.join(f'{t+1}:{100*pm[t,0]/max(pm[t,1],1):.1f}%'
                                for t in range(4, 59, 6)))
    print('saved', a.out)


if __name__ == '__main__':
    main()
