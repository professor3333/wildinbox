## Comparison (unseen development cameras)

| Model | Macro-F1 | Min species recall | Animal image recall | False-empty events @0.9 / @0.99 | Accepted @0.9 (precision) | Unsupported accepted | Seen-camera macro-F1 | CPU p50 ms | Train min | Selection rule |
|---|---|---|---|---|---|---|---|---|---|---|
| [baseline-frozen-effnetb0-logreg-v1](../baseline/README.md) | 0.258 | 0.100 | 83.4% | 164 / 127 | 728 @ 22.3% | 144 | 0.724 | 131 | - | reference |
| [finetune-e1-top-lossweight](../experiments/finetune-e1-top-lossweight/README.md) | 0.385 | 0.208 | 68.8% | 60 / 31 | 137 @ 75.9% | 14 | 0.760 | 130 | 17 | **pass**: macro-F1 gain +0.127; false-empty events 60 (limit 180); CPU latency 0.99x |
| [finetune-e2-deep-lossweight](../experiments/finetune-e2-deep-lossweight/README.md) | 0.403 | 0.124 | 73.4% | 54 / 26 | 204 @ 80.4% | 10 | 0.754 | 132 | 41 | **pass**: macro-F1 gain +0.145; false-empty events 54 (limit 180); CPU latency 1.00x |
| [finetune-e3-deep-balanced](../experiments/finetune-e3-deep-balanced/README.md) | 0.413 | 0.163 | 75.5% | 57 / 17 | 234 @ 69.7% | 34 | 0.747 | 132 | 70 | **pass**: macro-F1 gain +0.155; false-empty events 57 (limit 180); CPU latency 1.00x |
| [finetune-e4-deep-lossweight-photo](../experiments/finetune-e4-deep-lossweight-photo/README.md) | 0.383 | 0.083 | 71.4% | 49 / 32 | 222 @ 80.6% | 19 | 0.749 | 133 | 45 | **pass**: macro-F1 gain +0.125; false-empty events 49 (limit 180); CPU latency 1.01x |

Per-class recall:

| Model | empty | bobcat | cat | coyote | dog | opossum | rabbit | raccoon |
|---|---|---|---|---|---|---|---|---|
| baseline-frozen-effnetb0-logreg-v1 | 0.45 | 0.10 | 0.14 | 0.44 | 0.53 | 0.36 | 0.18 | 0.28 |
| finetune-e1-top-lossweight | 0.82 | 0.44 | 0.21 | 0.49 | 0.50 | 0.32 | 0.21 | 0.30 |
| finetune-e2-deep-lossweight | 0.85 | 0.50 | 0.12 | 0.51 | 0.54 | 0.38 | 0.32 | 0.28 |
| finetune-e3-deep-balanced | 0.69 | 0.49 | 0.16 | 0.42 | 0.57 | 0.44 | 0.43 | 0.39 |
| finetune-e4-deep-lossweight-photo | 0.78 | 0.51 | 0.08 | 0.53 | 0.50 | 0.36 | 0.34 | 0.29 |

False-empty events per camera at the reference thresholds:

| Model | 51 | 90 | 108 | 125 |
|---|---|---|---|---|
| baseline-frozen-effnetb0-logreg-v1 | 45 / 396 | 14 / 297 | 98 / 328 | 7 / 556 |
| finetune-e1-top-lossweight | 51 / 396 | 0 / 297 | 6 / 328 | 3 / 556 |
| finetune-e2-deep-lossweight | 47 / 396 | 1 / 297 | 3 / 328 | 3 / 556 |
| finetune-e3-deep-balanced | 43 / 396 | 6 / 297 | 1 / 328 | 7 / 556 |
| finetune-e4-deep-lossweight-photo | 47 / 396 | 0 / 297 | 0 / 328 | 2 / 556 |

Night vs day (unseen cameras):

| Model | Day macro-F1 | Night macro-F1 | Night false-empty |
|---|---|---|---|
| baseline-frozen-effnetb0-logreg-v1 | 0.233 | 0.231 | 121 / 1028 |
| finetune-e1-top-lossweight | 0.386 | 0.310 | 28 / 1028 |
| finetune-e2-deep-lossweight | 0.394 | 0.331 | 25 / 1028 |
| finetune-e3-deep-balanced | 0.394 | 0.356 | 19 / 1028 |
| finetune-e4-deep-lossweight-photo | 0.373 | 0.321 | 21 / 1028 |

**Selected by the rule:** finetune-e3-deep-balanced.
