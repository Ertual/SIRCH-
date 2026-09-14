# Retraining on grouped scene splits

This experiment retrains the complete 2x2 factorial on the sealed grouped splits
from `revision/exp11_grouped_scene_splits`:

- LSTM, principal training set;
- LSTM, augmented training set;
- GRU, principal training set;
- GRU, augmented training set.

The validation set contains 597 videos and is the only evaluation split available
to this training program. The test manifest is deliberately never loaded.

## Frozen-backbone optimization

EfficientNetB0 is frozen in the established SIRCH protocol. The script therefore
extracts its deterministic 20-frame representation once for the union of the
augmented training set and validation set, then trains the recurrent heads on
those cached tensors. The resulting head layers are shared with a complete SIRCH
model. Before export, every trained head is compared numerically with its complete
model on a sealed validation video; export is aborted unless the scores agree.

This removes repeated frozen-backbone computation without changing the network,
loss, optimizer, batch size, epoch limit, callbacks, frame sampling, or seed.

## Run

```powershell
C:\SIRCH_ENV\Scripts\python.exe revision\exp13_grouped_retraining\train_grouped_factorial.py
```

Large caches, checkpoints, weights, models, and live status files are written to
`C:\SIRCH_ENV\models\revision\exp13_grouped_retraining`. Final metadata and
publication artifacts are collected under this experiment's `outputs/` directory
after the four runs complete.
