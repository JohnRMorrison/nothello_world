# CCGP + Intervention results (OGPT vs interpretable models)

Generated 2026-08-13. Raw logs: pod `/workspace/nothello_world/ccgp_proper.log`
(CCGP linear); cluster `experiments/multi_intervention_nanda_L4/` and
`experiments/multi_intervention_cascade_L4/` (interventions).

## 1. CCGP linear matrix (N=100k, OGPT L6, `--ccgp-mode proper --probes linear`)

Gap = Within − CCGP (small = abstract/transferable). Chance = 0.5. All `null`
(random-split) Gaps ≈ 0 → estimators unbiased → Gaps trustworthy.

### Gap (Within − CCGP)
| mode | OGPT | H512_playedeven | H512_move_grid | H4096_playedeven | H4096_move_grid | J1B_svd2048 |
|---|---|---|---|---|---|---|
| null | 0.0000 | 0.0000 | −0.0002 | 0.0005 | −0.0001 | 0.0001 |
| phase_fwd | 0.0034 | 0.0542 | 0.0420 | 0.0185 | 0.1120 | 0.0144 |
| phase_bwd | 0.0810 | 0.1603 | 0.1919 | 0.2480 | 0.2682 | 0.2477 |
| recency_fixed | 0.0453 | 0.1029 | 0.0672 | 0.1216 | 0.0951 | 0.1051 |
| flip_true | 0.0491 | 0.6050 | 0.4325 | 0.6215 | 0.4021 | 0.4792 |
| crowd_frac | 0.1096 | 0.1490 | 0.1001 | 0.1426 | 0.1027 | 0.3449 |

### Within
| mode | OGPT | H512_playedeven | H512_move_grid | H4096_playedeven | H4096_move_grid | J1B_svd2048 |
|---|---|---|---|---|---|---|
| null | 0.9897 | 0.8273 | 0.8968 | 0.8139 | 0.8621 | 0.8417 |
| phase_fwd | 0.9847 | 0.7526 | 0.8225 | 0.7178 | 0.7738 | 0.7337 |
| phase_bwd | 0.9431 | 0.9154 | 0.9575 | 0.9040 | 0.8971 | 0.9479 |
| recency_fixed | 0.9726 | 0.7840 | 0.8188 | 0.8021 | 0.7868 | 0.7116 |
| flip_true | 0.9942 | 0.9005 | 0.9018 | 0.8966 | 0.8796 | 0.8763 |
| crowd_frac | 0.8969 | 0.8187 | 0.8590 | 0.7977 | 0.7659 | 0.8294 |

### flip_true as CCGP accuracy (chance = 0.5) — the headline
| | OGPT | H512_pe | H512_mg | H4096_pe | H4096_mg | J1B |
|---|---|---|---|---|---|---|
| flip CCGP | **0.945** | **0.296** | 0.469 | **0.275** | 0.478 | 0.397 |

**Verdict:** OGPT transfers "cell == color" across the flip condition almost
perfectly (0.945) — it tracks *current* color through captures (a real updating
world-model). The playedeven MLPs and J1B are **below chance** (0.28–0.40): their
color code is keyed to *who placed the disc*, so on captured (flipped) discs a
cross-condition decoder is systematically WRONG (anti-transfer / inversion). The
move_grid MLPs sit at chance (no placement-color cue → no transfer, no inversion).
Capacity does not fix it (H4096_pe flip 0.62 ≥ H512_pe 0.60). phase_bwd is a
distant second signal and worsens with capacity. J1B's signature is crowd (0.345).

(recency_fixed row RE-RUN at n=100k with the pos fix — see §3; OGPT 0.0453
confirms the standalone. The pre-fix buggy row was OGPT 0.155.)

## 1b. Deck-ready tables (Condition/Split × model)

Gap = Within − CCGP (small = abstract). `† ply-matched.` Columns: **OGPT** =
Othello-GPT L6; **512/4k · set/grid** = interpretable MLPs (H512/H4096 ×
move_set(playedeven)/move_grid); **J1B** = tree-bank (SVD-2048).

### Gap
| Condition | Split | OGPT | 512·set | 512·grid | 4k·set | 4k·grid | J1B |
|---|---|---|---|---|---|---|---|
| Random (control) | shuffled labels | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Phase forward | train 5–40 → 41–53 | 0.003 | 0.054 | 0.042 | 0.019 | 0.112 | 0.014 |
| Recency † | played ≤2 → >15 moves | 0.045 | 0.103 | 0.067 | 0.122 | 0.095 | 0.105 |
| Flip † | never → ever flipped | 0.049 | 0.605 | 0.433 | 0.622 | 0.402 | 0.479 |
| Phase backward | train 17–53 → 5–16 | 0.081 | 0.160 | 0.192 | 0.248 | 0.268 | 0.248 |
| Crowding † | nbrs <.25 → >.75 full | 0.109 | 0.149 | 0.100 | 0.143 | 0.103 | 0.345 |

### Within (companion — decodability ceiling within each condition)
| Condition | Split | OGPT | 512·set | 512·grid | 4k·set | 4k·grid | J1B |
|---|---|---|---|---|---|---|---|
| Random (control) | shuffled labels | 0.990 | 0.827 | 0.897 | 0.814 | 0.862 | 0.842 |
| Phase forward | train 5–40 → 41–53 | 0.985 | 0.753 | 0.823 | 0.718 | 0.774 | 0.734 |
| Recency † | played ≤2 → >15 moves | 0.973 | 0.784 | 0.819 | 0.802 | 0.787 | 0.712 |
| Flip † | never → ever flipped | 0.994 | 0.901 | 0.902 | 0.897 | 0.880 | 0.876 |
| Phase backward | train 17–53 → 5–16 | 0.943 | 0.915 | 0.958 | 0.904 | 0.897 | 0.948 |
| Crowding † | nbrs <.25 → >.75 full | 0.897 | 0.819 | 0.859 | 0.798 | 0.766 | 0.829 |

Within is high for every model on every condition (0.71–0.99) — the board IS
decodable within a condition. The story is entirely in the Gap: OGPT transfers
across conditions (esp. flip), the interpretable models don't. Note OGPT's Within
is uniformly ~0.94–0.99 (near-perfect board read at L6), while the interpretable
models sit ~0.71–0.96 — so their large flip Gaps are genuine transfer failures,
not a within-condition decodability artifact.

## 2. Intervention boundary margins (faithful Nanda vs faithful Li vs cd0)

- **Nanda** = single edit at resid-after-L4 (`--layer-intervene 5`), fixed L6
  probe, `--scale 2.0` (negate). `experiments/multi_intervention_nanda_L4`.
- **Li** = cascade Ls=4→final, per-layer native probes. `experiments/multi_intervention_cascade_L4`.
- **cd0** = single L5, per-cell calibrated. `experiments/multi_intervention_probs_cd0`.
- Nanda & Li are PAIRED (same positions: 4490 newly-legal / 4153 newly-illegal pts).

| metric | Nanda | Li (cascade) | cd0 |
|---|---|---|---|
| **Newly-legal** | | | |
| > most-probable illegal | 24.3% | 63.3% | 63.1% |
| median above illegal ceiling | −0.000 | +0.000 | +0.000 |
| within ±0.01 of illegal ceiling | 90.1% | 85.1% | 96.1% |
| < least-probable legal | 98.8% | 97.3% | 99.8% |
| median below legal floor | 0.094 | 0.088 | 0.088 |
| **Newly-illegal** | | | |
| below all legal | 49.4% | 54.9% | 44.9% |
| above all illegal | 85.0% | 89.9% | 99.6% |
| within ±0.02 of floor | 56.7% | 57.0% | 74.0% |
| ≥0.02 above floor (still-legal) | 3.2% | 9.4% | 2.5% |
| median vs legal floor | −0.000 (at) | +0.002 (below) | −0.002 (above) |

**Verdict:** single-layer (Nanda) is much weaker at *promotion* (newly-legal
clears the illegal ceiling only 24.3% vs Li's 63.3% on identical positions),
comparable/slightly stronger at *demotion* — exactly Li et al.'s argument that
one-layer edits are insufficient because later layers rebuild the state.
Neither promotes newly-legal cleanly into the legal band (~0.09 below floor).
Caveat: Nanda scale fixed at 2.0; a scale sweep would separate "single-layer
ceiling" from "underpowered edit."

Reference — paper intervention layers:
- **Li et al.** (2210.13382): intervene at EVERY layer Ls→final; best Ls=4 (5 layers).
- **Nanda** (neelnanda.io/.../othello): SINGLE edit at resid-after-layer-4 (L3 fallback),
  negate the layer-6 probe direction; probe trained at L6, transfers zero-shot to L4.

## 3. Recency off-by-one — DIAGNOSED, FIXED, VALIDATED

`sample_shared_positions` (matrix path) set `pos = nmoves` while board/residual
sit at token `t-1` and `place_step` is 0-indexed over `0..t-1`. So the current-ply
index is `t-1`, and `rec = pos − place_step` was uniformly +1 (no `rec=0` bucket,
i.e. the just-placed disc never appeared in the "recent" tail). Only `recency`
reads `rec`, so only that row diverged; phase/crowd/flip matched the standalone.

**Fix:** `pos = nmoves - 1` (compute_ccgp.py:304).

**Local validation (OGPT L6, match-ply, ≤2/≥15, n=30k):**
| variant | Gap |
|---|---|
| BUG reproduced (pos=nmoves) | 0.148  (≈ matrix's 0.155) |
| FIXED matrix (pos=nmoves-1) | 0.081 |
| standalone `ccgp_recency_extreme.py` | 0.079 |

Fixed matrix (0.081) ≈ standalone (0.079) → the two code paths now agree. The
earlier "~0.044" reference was a different-n/config number; the correct
matched-n OGPT recency Gap is ~0.08 (may shrink slightly at n=100k). OGPT is
NOT uniquely recency-dependent — that was the artifact.

**RE-RUN DONE (n=100k, fix applied).** Corrected recency_fixed row now in §1:
OGPT 0.0453 (matches the standalone 0.044 → fix validated at full n), 512·set
0.1029, 512·grid 0.0672, 4k·set 0.1216, 4k·grid 0.0951, J1B 0.1051. All fell vs
the buggy row. OGPT is the MOST recency-abstract; recency is a moderate, not
discriminating, condition (cf. flip).
