"""Visualize a hidden node's input heuristic on an 8x8 Othello board.

Each cell is divided into two diagonal halves:
  Top-left half  shows the EXCITATORY state of the cell
                 (the piece configuration that fires the node).
                 Background: dark green (Ex Core), medium green (Ex Aux),
                 white if the cell has no excitatory role.
  Bottom-right half shows the INHIBITORY state of the cell
                 (the configuration that silences the node).
                 Background: dark red (Inh Core), medium red (Inh Aux),
                 white if the cell has no inhibitory role.

Inside each half:
  filled black circle  = black piece
  filled white circle  = white piece
  blank                = cell is empty (not played)

Usage:
    python plot_node_inputs.py \
        --json node_heuristics_H512_playedeven.json \
        --parity odd --node 480 \
        --out node_input_odd_480.png
"""
import argparse
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle


def cell_name(c):
    return f"{chr(65 + c // 8)}{c % 8 + 1}"


DARK_GREEN = '#0b6e0b'
MID_GREEN = '#86c986'
DARK_RED = '#9b1c1c'
MID_RED = '#e89b9b'
NEUTRAL = '#ffffff'


def _draw_piece(ax, cx, cy, state, radius=0.18):
    """Draw a piece marker for a given state inside a triangular half.

    black  = solid black circle (a black piece on this cell)
    white  = solid white circle  (a white piece)
    empty  = open ring (outline only — the cell must be unplayed)
    """
    if state == 'black':
        ax.add_patch(Circle((cx, cy), radius,
                            facecolor='black', edgecolor='black', linewidth=0.5))
    elif state == 'white':
        ax.add_patch(Circle((cx, cy), radius,
                            facecolor='white', edgecolor='black', linewidth=0.8))
    elif state == 'empty':
        ax.add_patch(Circle((cx, cy), radius,
                            facecolor='none', edgecolor='black', linewidth=1.0))


def plot_input_heuristic(record, ax=None, title=None):
    sig = record['signature']

    # Per cell: (ex_color, ex_state) and (inh_color, inh_state).
    # If a role isn't present for the cell, leave it as (NEUTRAL, None).
    ex_role = {}  # cell -> (color, state)
    for c in sig['excitatory_sufficient_core']:
        ex_role[c['cell']] = (DARK_GREEN, c['state'])
    for c in sig['excitatory_sufficient_aux']:
        if c['cell'] not in ex_role:
            ex_role[c['cell']] = (MID_GREEN, c['state'])
    inh_role = {}
    for c in sig['inhibitory_sufficient_core']:
        inh_role[c['cell']] = (DARK_RED, c['state'])
    for c in sig['inhibitory_sufficient_aux']:
        if c['cell'] not in inh_role:
            inh_role[c['cell']] = (MID_RED, c['state'])

    involved_cells = set(ex_role) | set(inh_role)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # Draw 8x8 grid
    for r in range(8):
        for c in range(8):
            idx = r * 8 + c
            x0, y0 = c, 7 - r  # bottom-left of cell

            if idx in involved_cells:
                # Diagonal split (top-left triangle vs bottom-right triangle)
                # Top-left triangle vertices: (x0,y0+1), (x0+1,y0+1), (x0,y0)
                # Bottom-right triangle vertices: (x0+1,y0+1), (x0+1,y0), (x0,y0)
                ex_color, ex_state = ex_role.get(idx, (NEUTRAL, None))
                inh_color, inh_state = inh_role.get(idx, (NEUTRAL, None))
                ax.add_patch(Polygon([(x0, y0 + 1), (x0 + 1, y0 + 1), (x0, y0)],
                                     facecolor=ex_color, edgecolor='black',
                                     linewidth=0.4))
                ax.add_patch(Polygon([(x0 + 1, y0 + 1), (x0 + 1, y0), (x0, y0)],
                                     facecolor=inh_color, edgecolor='black',
                                     linewidth=0.4))
                # Pieces inside each half.  Centroids of the two triangles.
                if ex_state is not None:
                    _draw_piece(ax, x0 + 1/3, y0 + 2/3, ex_state)
                if inh_state is not None:
                    _draw_piece(ax, x0 + 2/3, y0 + 1/3, inh_state)
            else:
                # Plain empty cell
                ax.add_patch(Rectangle((x0, y0), 1, 1,
                                       facecolor=NEUTRAL,
                                       edgecolor='black', linewidth=0.4))

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    if title is None:
        title = (f"Node {record['parity']}-{record['node']}  "
                 f"({record['classification']}, "
                 f"bias = {sig['bias']:+.2f})")
    ax.set_title(title, fontsize=11)

    # Legend
    legend_elems = [
        Rectangle((0, 0), 1, 1, facecolor=DARK_GREEN, label='Ex Core'),
        Rectangle((0, 0), 1, 1, facecolor=MID_GREEN, label='Ex Aux'),
        Rectangle((0, 0), 1, 1, facecolor=DARK_RED, label='Inh Core'),
        Rectangle((0, 0), 1, 1, facecolor=MID_RED, label='Inh Aux'),
    ]
    ax.legend(handles=legend_elems, loc='center left',
              bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True, help='Path to node_heuristics JSON')
    ap.add_argument('--parity', choices=['even', 'odd'], required=True)
    ap.add_argument('--node', type=int, required=True)
    ap.add_argument('--out', default=None, help='Output PNG; if omitted, displays')
    args = ap.parse_args()

    with open(args.json) as f:
        nodes = json.load(f)
    target = None
    for n in nodes:
        if n['parity'] == args.parity and n['node'] == args.node:
            target = n
            break
    if target is None:
        sys.exit(f"No node {args.parity}-{args.node} in {args.json}")

    fig = plot_input_heuristic(target)
    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=140, bbox_inches='tight')
        print(f"Saved {args.out}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
