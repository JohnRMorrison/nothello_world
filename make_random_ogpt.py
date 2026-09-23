"""Save a randomly initialised Othello-GPT: the untrained control.

Same architecture as the trained model, weights straight from mingpt's
initialiser and never trained -- which is what Li et al.'s `--random` flag does
(train_probe_othello.py: model.apply(model._init_weights)).

Written to disk as a normal checkpoint so every downstream script that takes a
--ckpt (train_nanda_probe_extended.py, analyze_nanda_probe_per_cell.py, ...)
can use it unmodified.

Usage:
    python make_random_ogpt.py --out ckpts/gpt_random_init.ckpt --seed 0
"""
import argparse
import torch

from mingpt.model import GPT, GPTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='ckpts/gpt_random_init.ckpt')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    cfg = GPTConfig(61, 59, n_layer=8, n_head=8, n_embd=512)
    model = GPT(cfg)                     # GPT.__init__ applies _init_weights
    sd = model.state_dict()
    torch.save(sd, a.out)

    n = sum(p.numel() for p in model.parameters())
    tok, pos = sd['tok_emb.weight'], sd['pos_emb']
    print(f'wrote {a.out}  (seed {a.seed}, {n:,} parameters)')
    print(f'  tok_emb std {tok.std():.4f}   (trained model: 0.153)')
    print(f'  pos_emb std {pos.std():.4f}   (init is exactly 0; trained: 0.136)')


if __name__ == '__main__':
    main()
