#!/usr/bin/env bash
# Fetch the OrganMNIST3D source file and verify it against the
# medmnist package's published MD5 before any use.
# OrganMNIST3D is part of MedMNIST v2 (Yang et al., Scientific Data
# 2023), distributed via Zenodo record 10519652 under CC BY 4.0.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
curl -L -o data/organmnist3d.npz \
  "https://zenodo.org/records/10519652/files/organmnist3d.npz?download=1"
echo "a0c5a1ff56af4f155c46d46fbb45a2fe  data/organmnist3d.npz" | md5sum -c -
sha256sum data/organmnist3d.npz
# Expected SHA-256 (computed in the foreground at build time, 2026-10-29):
# f99ea411e1c9422f28524b493f2ca74e8781a5dda878d9d5481be97063114bdc
