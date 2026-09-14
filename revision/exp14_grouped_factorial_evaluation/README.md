# Grouped factorial evaluation

This experiment is the post-training protocol for the four models retrained on
the scene-grouped splits.

The runner enforces the following order:

1. load the four completed best-weight exports from exp13;
2. score the sealed 597-video validation split and select theta/K independently
   for each model using the established grid and tie-break rule;
3. seal the four selected configurations by SHA-256;
4. only then read and score the sealed 597-video test split once;
5. run paired McNemar and 10,000-iteration paired bootstrap comparisons;
6. estimate the architecture by enrichment interaction;
7. evaluate the 30 sealed hard-negative videos with no retuning.

Run after exp13 reports `status: complete`:

```powershell
C:\SIRCH_ENV\Scripts\python.exe revision\exp14_grouped_factorial_evaluation\run_grouped_evaluation.py
```

`queue_after_training.ps1` can be started while exp13 is active. It waits without
touching the training process and invokes the evaluator only after exp13 records a
normal completion.

The scoring cache supports interruption recovery. A completed final-test result
is never recomputed.
