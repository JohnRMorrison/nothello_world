"""Accuracy of linear board probes (mode 0, empty/yours/mine) per layer, on
held-out synthetic games, under every model checkpoint given.

Checks, for each (checkpoint, probe) pair:
  * overall accuracy over all 64 squares
  * accuracy on empty squares and on occupied squares (mine vs yours)
  * occupied accuracy with mine/yours SWAPPED (sanity: must be much lower,
    otherwise the class convention is reversed)
  * accuracy split by position parity (mode 0 is trained on one parity only)
for two position ranges: the probe training range [5, 59) and the
Section 5.4 sampling range [10, 50).

Labels follow the intervention code (generate_paper_numbers.bv2cls):
mine = board.next_hand_color after the prefix.

Parity columns use the absolute move index t. Mode 0 was trained on slice
[0::2] starting at t=5, i.e. on ODD t only, so occOdd is in-distribution and
occEven measures generalisation to the other player's turn.

Layer convention: --probe path:K means the probe reads resid_post of mingpt
block K (TransformerLens resid_post K), i.e. after K+1 blocks.

Usage:
    python check_probe_accuracy.py
    python check_probe_accuracy.py --probe mechanistic_interpretability/main_linear_probe_L3.pth:3
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState

sys.path.insert(0, 'experiments/mathematical_transformation_experiments')
from probe_state_pred_for_othello import (  # noqa: E402
    tokenize_games, SYNTHETIC_DIR, GAME_LEN, VOCAB_SIZE,
)

EMPTY, YOURS, MINE = 0, 1, 2
DEFAULT_PROBES = [
    'mechanistic_interpretability/main_linear_probe_L4.pth:4',
    'mechanistic_interpretability/main_linear_probe_L5.pth:5',
    'mechanistic_interpretability/main_linear_probe.pth:6',
    'mechanistic_interpretability/main_linear_probe_L7.pth:7',
    'mechanistic_interpretability/main_linear_probe_L3.pth:3',
]
DEFAULT_CKPTS = ['ckpts/gpt_nanda_synthetic.ckpt', 'ckpts/gpt_synthetic.ckpt']
RANGES = {'train_5_59': (5, 59), 'sec54_10_50': (10, 50)}


def load_model(path, device):
    cfg = GPTConfig(vocab_size=VOCAB_SIZE, block_size=59,
                    n_layer=8, n_head=8, n_embd=512)
    model = GPT(cfg)
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    res = model.load_state_dict(state, strict=False)
    real_missing = [k for k in res.missing_keys if not k.endswith('.mask')]
    if real_missing:
        raise RuntimeError(f'{path}: missing weights {real_missing[:5]}')
    print(f'  loaded {path}  missing={res.missing_keys[:3]}  '
          f'unexpected={res.unexpected_keys[:3]}')
    return model.to(device).eval()


def labels_for_game(game):
    """(GAME_LEN, 8, 8) class labels after each move, mine = next_hand_color."""
    board = OthelloBoardState()
    out = np.full((GAME_LEN, 8, 8), -1, dtype=np.int8)
    for t, m in enumerate(game):
        board.umpire(m)
        s = board.state
        col = board.next_hand_color
        lab = np.where(s == 0, EMPTY, np.where(s == col, MINE, YOURS))
        out[t] = lab
    return out


@torch.no_grad()
def resid_all_layers(model, tokens):
    """List of resid_post for blocks 0..7, each (B, T, 512)."""
    b, t = tokens.size()
    x = model.drop(model.tok_emb(tokens) + model.pos_emb[:, :t, :])
    outs = []
    for block in model.blocks:
        x = block(x)
        outs.append(x)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', default=None)
    ap.add_argument('--probe', action='append', default=None,
                    help='path:block_index (repeatable)')
    ap.add_argument('--n-games', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=200)
    ap.add_argument('--out', default='Intervention Results/probe_accuracy.txt')
    args = ap.parse_args()
    ckpts  = args.ckpt or DEFAULT_CKPTS
    probes = args.probe or DEFAULT_PROBES

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    files = sorted(f for f in os.listdir(SYNTHETIC_DIR) if f.endswith('.pickle'))
    eval_file = files[-1]
    print(f'{len(files)} synthetic files; probe training uses the first 2 '
          f'({files[:2]}); evaluating on the LAST: {eval_file}')
    import pickle
    with open(os.path.join(SYNTHETIC_DIR, eval_file), 'rb') as fh:
        games = [g for g in pickle.load(fh) if len(g) == GAME_LEN]
    games = games[:args.n_games]
    print(f'{len(games)} full-length eval games')

    labels = torch.from_numpy(np.stack([labels_for_game(g) for g in games]))
    tokens = tokenize_games(games)[:, :-1]          # (G, 59), like training
    labels = labels[:, :tokens.shape[1]]            # align to 59 positions
    T = tokens.shape[1]
    t_idx = torch.arange(T)

    loaded = []
    for spec in probes:
        path, layer = spec.rsplit(':', 1)
        if not os.path.exists(path):
            print(f'  SKIP (not found): {path}')
            continue
        p = torch.load(path, map_location='cpu').detach().float()
        loaded.append((path, int(layer), p[0].to(device)))   # mode 0
    if not loaded:
        sys.exit('no probes found')

    lines = []
    def emit(s=''):
        print(s); lines.append(s)

    emit(f'Probe accuracy (mode 0: empty/yours/mine), eval file {eval_file}, '
         f'{len(games)} games')
    hdr = (f'{"checkpoint":28s} {"probe":42s} {"blk":>3s} {"range":12s} '
           f'{"all":>6s} {"empty":>6s} {"occ":>6s} {"occSWAP":>7s} '
           f'{"occEven":>7s} {"occOdd":>7s}')

    for ck in ckpts:
        if not os.path.exists(ck):
            emit(f'SKIP checkpoint (not found): {ck}')
            continue
        model = load_model(ck, device)
        # counts[(probe_idx, range)] = dict of [correct, total]
        counts = {}
        for i in range(0, len(games), args.batch):
            tok = tokens[i:i + args.batch].to(device)
            lab = labels[i:i + args.batch].to(device).long()     # (B,T,8,8)
            res = resid_all_layers(model, tok)
            for pi, (path, layer, W) in enumerate(loaded):
                pred = torch.einsum('btd,drco->btrco', res[layer], W).argmax(-1)
                swap = torch.where(lab == MINE, YOURS,
                                   torch.where(lab == YOURS, MINE, lab))
                for rname, (lo, hi) in RANGES.items():
                    sel = (t_idx >= lo) & (t_idx < hi)
                    sel = sel.to(device)[None, :, None, None].expand_as(lab)
                    even = (t_idx % 2 == 0).to(device)[None, :, None, None].expand_as(lab)
                    occ = lab != EMPTY
                    c = counts.setdefault((pi, rname), {})
                    def acc(key, mask, target):
                        a = c.setdefault(key, [0, 0])
                        a[0] += int(((pred == target) & mask).sum())
                        a[1] += int(mask.sum())
                    acc('all',     sel,                lab)
                    acc('empty',   sel & ~occ,         lab)
                    acc('occ',     sel & occ,          lab)
                    acc('occSWAP', sel & occ,          swap)
                    acc('occEven', sel & occ & even,   lab)
                    acc('occOdd',  sel & occ & ~even,  lab)
        emit(); emit(hdr); emit('-' * len(hdr))
        for pi, (path, layer, _) in enumerate(loaded):
            for rname in RANGES:
                c = counts[(pi, rname)]
                f = lambda k: c[k][0] / max(c[k][1], 1)
                emit(f'{os.path.basename(ck):28s} {os.path.basename(path):42s} '
                     f'{layer:3d} {rname:12s} {f("all"):6.3f} {f("empty"):6.3f} '
                     f'{f("occ"):6.3f} {f("occSWAP"):7.3f} '
                     f'{f("occEven"):7.3f} {f("occOdd"):7.3f}')
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'\nSaved -> {args.out}')


if __name__ == '__main__':
    main()
