#!/bin/bash
# Reinstall Python deps on a fresh RunPod container.
#
# The /workspace network volume (othello_world) persists the repo, data,
# ckpts, and feature chunks across pods -- only the system pip packages live on
# the ephemeral container disk, so they must be reinstalled each new container.
#
# Usage on a new pod (volume mounted at /workspace):
#   bash /workspace/nothello_world/pod_setup.sh
#
# Then run everything with `python3.13` (NOT bare `python`, which is a different
# interpreter that can't see these packages).
set -e

echo "== installing deps into python3.13 =="
pip install --quiet numpy scikit-learn tqdm
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu

echo "== verify =="
python3.13 -c "import sys, torch, numpy, sklearn, tqdm; \
print('python', sys.executable); \
print('torch', torch.__version__, '| numpy', numpy.__version__, '| sklearn', sklearn.__version__)"

echo "== done. Run scripts with: python3.13 <script>.py =="
