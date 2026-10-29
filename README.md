---
license: mit
pipeline_tag: image-classification
language: en
tags:
  - medical-imaging
  - medmnist
  - organmnist3d
  - 3d-cnn
  - pytorch
  - onnx
  - safetensors
---

# OrganCNN — OrganMNIST3D 11-class organ classifier

**A compact 3D CNN (105,307 parameters), designed and trained from scratch for multi-class classification of 28×28×28 CT bounding-box volumes into 11 body organs, released with something most model uploads don't have: a script that re-derives every headline metric from the shipped weights and data, so the numbers below are checkable, not just claimed.**

Read the weak spots plainly, because they are the interesting part: **the model's dominant failure is left/right anatomy.** Of its 87 test errors, 53 (61%) are confusions between the left and right member of an organ pair — 26 femur-left volumes read as femur-right (femur-left recall 0.5692, the weakest class) and 20 kidney-right volumes read as kidney-left (recall 0.6471). The training run deliberately used **no flip/permutation augmentation** — flipping a volume changes the correct laterality label — yet the 28³ bounding boxes still carry too little orientation context for the model to reliably tell left from right. Both lungs, by contrast, are classified perfectly (21/21 each). It is the portfolio's first multi-class 3D model and its second volumetric one (after VesselCNN).

> **Research and benchmark use only. This is a 28×28×28 benchmark model trained on downsampled CT bounding boxes. It is not a medical device, it has never been evaluated on clinical CT data, and it must not be used for diagnosis or treatment planning.**

## Verified results — OrganMNIST3D test split (n = 610)

| Metric | Value | Source |
|---|---|---|
| Accuracy | **0.8574** (523/610) | reproduced 2026-10-29 |
| Macro AUC (one-vs-rest) | **0.9860** | reproduced 2026-10-29 |
| Balanced accuracy | 0.8780 | reproduced 2026-10-29 |
| Majority-class baseline | 0.1131 | reproduced 2026-10-29 |
| Best validation balanced accuracy | 0.9673 (epoch 24) | recorded at training time |
| Best validation AUC (macro OvR) | 0.9994 (epoch 24) | recorded at training time |

Note the validation–test gap (validation accuracy 0.9689 vs. test 0.8574): the validation split comes from the source dataset's training scans while the test split comes from different source scans, and the checkpoint was selected on the validation split. The gap is reported here, not smoothed over; read the test numbers as the headline.

Per-class results on the test split (all reproduced):

| Class | Recall | Precision | Support |
|---|---|---|---|
| liver | 0.9855 (68/69) | 0.9714 | 69 |
| kidney-right | **0.6471 (44/68)** | 0.9362 | 68 |
| kidney-left | 0.8261 (57/69) | **0.6706** | 69 |
| femur-right | 0.9231 (60/65) | **0.6522** | 65 |
| femur-left | **0.5692 (37/65)** | 0.9024 | 65 |
| bladder | 0.9091 (60/66) | 0.9231 | 66 |
| heart | 0.9286 (26/28) | 1.0000 | 28 |
| lung-right | 1.0000 (21/21) | 1.0000 | 21 |
| lung-left | 1.0000 (21/21) | 1.0000 | 21 |
| spleen | 0.9130 (63/69) | 0.8750 | 69 |
| pancreas | 0.9565 (66/69) | 0.9429 | 69 |

The failure analysis (full record in `assets/evaluation_results.json`) is dominated by one pattern: **laterality**. The top confusion pairs are femur-left → femur-right (26 of 65 femur-left volumes; 40% of the class), kidney-right → kidney-left (20 of 68; 29%), then a small spleen ↔ kidney-left exchange (6 + 5). Left/right pair confusions total 53 of 87 errors. Errors are also mostly *unconfident* — mean predicted probability 0.4882 on errors vs. 0.7801 on correct predictions — with one loud exception in the packaged samples: a bladder volume read as liver at 0.814 confidence. The first five misclassified test volumes are all laterality errors.

Context, not a claim of parity: the MedMNIST v2 paper's reference ResNet-18 (3D) reports AUC 0.996 / ACC 0.907 on this dataset — a model roughly 300× this one's size. OrganCNN (0.9860 / 0.8574) does not match it; the gap, and where it concentrates (laterality), is documented rather than hidden.

## Quickstart

```bash
pip install "git+https://github.com/habib-analyst/organcnn.git"
```

```python
from organcnn import load_model, predict

model = load_model("model.safetensors")
result = predict(model, "assets/samples/most_confident_correct_74.npy")
# {'label': 'bladder', 'confidence': 0.998, 'probabilities': {...}}
```

Inputs are single volumes: a `.npy` path or a NumPy array of shape (28, 28, 28), uint8 as released. Preprocessing contract (identical to training): /255 → normalize with the OrganMNIST3D train-split mean/std (**0.50092 / 0.28022**, scalar — the volumes are single-channel). Labels are the MedMNIST v2 OrganMNIST3D order: 0 liver, 1 kidney-right, 2 kidney-left, 3 femur-right, 4 femur-left, 5 bladder, 6 heart, 7 lung-right, 8 lung-left, 9 spleen, 10 pancreas (see `config.json`).

ONNX: `organcnn.onnx` takes the same normalized `(N, 1, 28, 28, 28)` float tensor and returns logits.

## Verify it yourself

```bash
python examples/verify_metrics.py
```

This re-evaluates the shipped weights on the bundled test split (`assets/organmnist3d_test.npz`) and compares against the training record. On 2026-10-29 it printed `ALL HEADLINE METRICS REPRODUCED` — accuracy, macro AUC, balanced accuracy, the full 11×11 confusion matrix, and all per-class recalls matched exactly.

```bash
python examples/predict_demo.py   # packaged samples + first misclassifications
python -m pytest -q               # 8 tests: formats agree, metrics files agree, parity
```

The demo is honest about the weak spots by construction: the four packaged samples were selected by a fixed rule (see `assets/samples/index.json`) that includes the first misclassified volume (a femur-left read as femur-right) and the most confident error (a bladder read as liver at 0.814). Of the four, two are classified correctly and two are errors, and the demo prints all four.

## Model details

- **Architecture:** 3 conv blocks (16→32→64 channels, 3×3×3 convolutions, BatchNorm, ReLU, max-pool, dropout) + global-average-pooling head with a single linear layer — the same compact family as the sibling SonoCNN / PathoCNN / VesselCNN releases, with an 11-way head. The GAP head is deliberate: a flattened MLP head would add ~1.4M parameters against only 971 training volumes. 105,307 trainable parameters; 0.4 MB in safetensors.
- **Training:** from scratch on the OrganMNIST3D train split (971 volumes), AdamW (lr 2e-3, weight decay 1e-4, OneCycle cosine over 30 epochs), batch 32, seed 42. Loss: class-weighted cross-entropy with weights N/(11·n_class) — from 0.7676 for the five 115-volume classes to 2.2634 for the two 39-volume lung classes — because heart (40) and the lungs (39 each) have a third of the data of the largest classes. All 30 epochs ran; the shipped weights are the epoch-24 checkpoint, selected by best validation balanced accuracy (ties by macro OvR AUC). Summed epoch time was 3,898 s (~65 min) on a 2-core CPU; the run was interrupted once by the environment after epoch 7 and resumed losslessly from the per-epoch checkpoint (see `training_log.txt`).
- **Augmentation: none — deliberately.** The VesselCNN sibling trains with octahedral (flip/permutation) augmentation because its label is a local shape property. Here, several labels *are* orientation: a flipped kidney-right volume is a kidney-left volume. Using that augmentation without a label remap would train on corrupted labels, so this run uses no geometric augmentation at all. The irony is recorded in the failure analysis: even without augmentation corrupting laterality, left/right confusion is still the dominant error — the bounding boxes themselves carry weak orientation signal at 28³.

## What OrganCNN is not

- **Not a clinical tool.** It classifies 28³ benchmark bounding boxes derived from CT scans, not patients or scans — and it cannot reliably tell a left femur from a right one. Nothing about it transfers to a diagnostic workflow.
- **Not state of the art, or close.** The reference ResNet-18 (3D) scores AUC 0.996 / ACC 0.907 on this dataset (above); this model's 0.9860 / 0.8574 trails it, with the deficit concentrated in two classes. The release exists because its numbers are verified and its failure modes are documented, not because it tops a leaderboard.
- **Not laterality-safe.** 61% of its errors swap left and right. Any use that cares which side an organ is on — which is most conceivable uses — is out of scope for these weights.
- **Not trained on private or clinical data.** Only the public OrganMNIST3D benchmark (below).
- **Confidences are not calibrated.** Softmax outputs are rankings, not probabilities of being right: the most confident error in the test split scores 0.814, and correct predictions average only 0.7801.

## Data and attribution

OrganMNIST3D is part of **MedMNIST v2** (Yang et al., *MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification*, Scientific Data 2023). Its source is the same as that of the Organ{A,C,S}MNIST datasets: 3D CT images from the **Liver Tumor Segmentation Benchmark (LiTS)**, with bounding-box annotations of 11 body organs from another study providing the labels; Hounsfield-Unit values were windowed to gray-scale, and OrganMNIST3D uses the 3D bounding boxes directly, resized to 28×28×28. The dataset is licensed **CC BY 4.0**; `assets/organmnist3d_test.npz` (the official test split only) is redistributed here for metric reproduction, with attribution. The source file used for training is Zenodo record 10519652, md5 `a0c5a1ff56af4f155c46d46fbb45a2fe` (full data card: `DATA.md`). Split hygiene was verified in the foreground at build time: zero exact-duplicate volumes across splits, and the splits derive from disjoint source CT scans (115/16/70) by the dataset authors' construction. The MIT license of this repository covers the code and the trained weights, not the dataset.

## Files

| File | Contents |
|---|---|
| `model.safetensors` | Release weights (safetensors) |
| `organcnn.onnx` | ONNX export (logits) |
| `config.json` | Architecture + preprocessing config |
| `DATA.md` | Dataset card: source, hashes, splits, class balance, caveats |
| `training_log.txt` | Per-epoch training log, including the interruption/resume record |
| `src/organcnn/` | `model.py` (architecture), `infer.py` (loading + prediction API) |
| `scripts/train.py` | The exact training script that produced the weights (resumable) |
| `scripts/evaluate.py` | Test evaluation + failure analysis + sample packaging |
| `scripts/export_onnx.py` | ONNX export + parity check |
| `scripts/download_data.sh` | Dataset fetch with MD5 verification |
| `examples/predict_demo.py` | Packaged samples + first misclassifications, printed honestly |
| `examples/verify_metrics.py` | Full metric reproduction from the artifacts |
| `assets/organcnn.pt` | Training checkpoint (state dict + selection record) |
| `assets/organcnn_metrics.json` | Training record + test metrics |
| `assets/evaluation_results.json` | Reproduced test metrics + full failure analysis |
| `assets/organmnist3d_test.npz` | OrganMNIST3D official test split (CC BY 4.0, see attribution) |
| `assets/samples/` | Four test volumes (.npy) + `index.json` with measured predictions |

## License

MIT for code and weights (see `LICENSE`). Dataset under CC BY 4.0 as above.
