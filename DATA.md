# DATA.md — dataset card for OrganCNN

## Source (the only source)

**OrganMNIST3D** — part of **MedMNIST v2** (Yang et al., *MedMNIST v2 —
A large-scale lightweight benchmark for 2D and 3D biomedical image
classification*, Scientific Data 2023). The source of OrganMNIST3D is
the same as that of the Organ{A,C,S}MNIST datasets: 3D computed
tomography (CT) images from the **Liver Tumor Segmentation Benchmark
(LiTS)**, with bounding-box annotations of 11 body organs from another
study used to obtain the organ labels. Hounsfield-Unit (HU) values were
transformed into gray-scale with an abdominal window. Instead of
cropping 2D center slices, OrganMNIST3D directly uses the 3D bounding
boxes, processed into 28×28×28 volumes. The same 115 and 16 CT scans
from the source training set are used as the training and validation
sets, respectively, and the same 70 CT scans from the source test set
are treated as the test set (per the `medmnist` package INFO).

- Official distribution used: Zenodo record 10519652 (the MedMNIST v2
  release file), fetched 2026-10-29 (Asia/Karachi) with
  `scripts/download_data.sh` (curl; the fetch is MD5-checked before
  use). The full file is redistributed in neither repository; the
  official **test split only** is bundled under `assets/` so the
  headline metrics can be re-derived from the release artifacts.
- **License: CC BY 4.0** (the `medmnist` package's dataset INFO and
  the MedMNIST v2 paper). Attribution: this card and the README
  citation. The MIT license of this repository covers the code and
  the trained weights, not the dataset.
- No other dataset was used. No private or clinical data beyond the
  public benchmark release.

## Exact file used (verified in the foreground at build time)

| File | Bytes | MD5 | SHA-256 |
|---|---|---|---|
| `organmnist3d.npz` | 32,657,349 | `a0c5a1ff56af4f155c46d46fbb45a2fe` | `f99ea411e1c9422f28524b493f2ca74e8781a5dda878d9d5481be97063114bdc` |

The MD5 equals the value published in the `medmnist` package's INFO
for this dataset; the SHA-256 was computed in the foreground from the
downloaded file.

## Splits, from the data itself

Verified by loading the file (not copied from documentation):

| Split | Volumes | Majority class | Majority baseline |
|---|---|---|---|
| train | 971 | liver / kidney-right / kidney-left / spleen / pancreas (115 each) | 0.1184 |
| val | 161 | liver / kidney-right / kidney-left / spleen / pancreas (16 each) | 0.0994 |
| test | 610 | liver / kidney-left / spleen / pancreas (69 each) | 0.1131 |

Per-class counts (official label order):

| Class | Train | Val | Test |
|---|---|---|---|
| liver | 115 | 16 | 69 |
| kidney-right | 115 | 16 | 68 |
| kidney-left | 115 | 14 | 69 |
| femur-right | 91 | 14 | 65 |
| femur-left | 93 | 15 | 65 |
| bladder | 94 | 15 | 66 |
| heart | 40 | 13 | 28 |
| lung-right | 39 | 13 | 21 |
| lung-left | 39 | 13 | 21 |
| spleen | 115 | 16 | 69 |
| pancreas | 115 | 16 | 69 |

- Volumes are uint8, shape (28, 28, 28), values in [0, 255]; labels
  are uint8 in {0, …, 10} with the order in the table above.
- **Exact-duplicate check:** MD5 hashes of every volume were compared
  across splits in the foreground — zero volumes appear in more than
  one split, and every volume is unique within its own split. (This
  checks exact duplicates only. The splits were made by the dataset
  authors at the source-scan level — 115/16/70 CT scans — as
  documented in the package INFO, so volumes in different splits come
  from different source scans by construction of the source split.)
- **Density:** unlike the sparse VesselMNIST3D segments, these are
  dense CT bounding boxes: 92.10% of train voxels are non-zero.

## Preprocessing and normalization

- The only preprocessing is the contract in `organcnn.infer`:
  /255, then normalize with the train-split scalar mean/std computed
  with chunked float64 accumulation over all 971 train volumes:
  exact mean 0.50092, exact std 0.28022 (these are also the rounded
  contract constants; `scripts/train.py` asserts agreement to 1e-4).
- No resampling, cropping, or intensity clipping: the volumes are
  used exactly as released.
- **No training augmentation is used.** Several labels encode
  left/right anatomy (kidney-right vs kidney-left, femur-right vs
  femur-left, lung-right vs lung-left), so flips and axis
  permutations — used for the VesselCNN sibling, whose label is a
  local shape property — would change the correct label here unless
  paired with a label remapping. This run takes the simpler,
  auditable option: no geometric augmentation. Recorded in
  `scripts/train.py` and the README.
- Class weighting in the loss (weights N/(11·n_class), from
  [0.7676 ×5 classes at n=115] to [2.2634 for the two lung classes at
  n=39]) is a training decision recorded in
  `assets/organcnn_metrics.json`, included here because it is part of
  how the smaller heart/lung classes were handled.
