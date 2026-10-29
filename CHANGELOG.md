# Changelog

## 0.1.0 - 2026-10-29

- Initial release of OrganCNN as a standalone model package — the
  portfolio's first multi-class 3D model and second volumetric model
  (after VesselCNN).
- From-scratch OrganMNIST3D model (105,307 parameters) trained for
  this release: all 30 planned epochs on the official 971-volume
  train split, seed 42, class-weighted cross-entropy, **no geometric
  augmentation** (laterality labels make flips/permutations
  label-changing — the deliberate difference from the VesselCNN
  recipe), checkpoint selected by validation balanced accuracy
  (epoch 24; val balanced accuracy 0.9673, val macro OvR AUC 0.9994).
  The run was interrupted once by the environment after epoch 7 and
  resumed losslessly from the per-epoch checkpoint. Training script
  included (`scripts/train.py`, resumable per epoch).
- Verified weights in three formats: `model.safetensors`, the
  training checkpoint (`assets/organcnn.pt`), and ONNX
  (`organcnn.onnx`, parity vs PyTorch measured at 2.38e-06 max logit
  difference over 32 test volumes, argmax agreement on all 32).
- Every headline metric reproduced from the shipped artifacts
  (`examples/verify_metrics.py`): test accuracy 0.8574, macro OvR
  AUC 0.9860, balanced accuracy 0.8780, full 11×11 confusion matrix,
  per-class recalls — exact match with the training record. The weak
  spots are documented on the card, not hidden: femur-left recall
  0.5692 and kidney-right recall 0.6471 are the weakest classes, and
  53 of 87 test errors (61%) are left/right pair confusions.
- Failure analysis included (`assets/evaluation_results.json`):
  ranked confusion pairs, the complete error inventory with
  predicted probabilities, and correct-vs-incorrect confidence means.
- Inference API (`load_model`, `predict`, `predict_proba`,
  `preprocess`) with the exact training preprocessing contract.
- Demo (`examples/predict_demo.py`): the four packaged sample
  volumes — selected by a fixed rule that includes the first error
  and the most confident error — plus the first misclassified test
  volumes.
