# Fine-tuning experiments: results and decision

Four EfficientNet-B0 fine-tuning experiments compared with the frozen-embedding
baseline on the **unseen-camera development partitions** (calibration cameras
51, 108 and policy-validation cameras 90, 125). The locked final test
(9 cameras) was not opened.

- Full comparison table: [`comparison.md`](comparison.md) (`uv run wildinbox compare`)
- Per-model reports: [baseline](../baseline/README.md) ·
  [E1](finetune-e1-top-lossweight/README.md) ·
  [E2](finetune-e2-deep-lossweight/README.md) ·
  [E3](finetune-e3-deep-balanced/README.md) ·
  [E4](finetune-e4-deep-lossweight-photo/README.md)
- Selection rule, committed before any fine-tuning result existed:
  [`configs/experiments/selection.yaml`](../../configs/experiments/selection.yaml)

## Experiments

Every experiment starts from torchvision ImageNet-1k weights, trains 3 epochs on
the training partition only (20,287 fit images, 7 cameras), and uses box-safe
crops plus horizontal flips. Each later experiment changes one factor.

| Id | Trainable layers | Class imbalance | Extra augmentation | Config |
|---|---|---|---|---|
| E1 | blocks 6-8 + head | class-weighted loss | - | [`finetune-e1-top-lossweight.yaml`](../../configs/experiments/finetune-e1-top-lossweight.yaml) |
| E2 | blocks 3-8 + head | class-weighted loss | - | [`finetune-e2-deep-lossweight.yaml`](../../configs/experiments/finetune-e2-deep-lossweight.yaml) |
| E3 | blocks 3-8 + head | class-balanced sampling | - | [`finetune-e3-deep-balanced.yaml`](../../configs/experiments/finetune-e3-deep-balanced.yaml) |
| E4 | blocks 3-8 + head | class-weighted loss | photometric | [`finetune-e4-deep-lossweight-photo.yaml`](../../configs/experiments/finetune-e4-deep-lossweight-photo.yaml) |

## Results (unseen cameras)

Scores are uncalibrated. Event-level numbers use the reference thresholds
P(empty) >= 0.9 and species >= 0.9, which are reference points, not chosen
operating points.

| Model | Macro-F1 | Min species recall | Animal events filtered as empty | Accepted labels (precision) | Unsupported accepted as known | CPU p50 ms | Selection rule |
|---|---|---|---|---|---|---|---|
| Baseline | 0.258 | 0.100 | 164 / 1577 | 728 (22.3%) | 144 | 131 | reference |
| E1 | 0.385 | 0.208 | 60 / 1577 | 137 (75.9%) | 14 | 130 | pass |
| E2 | 0.403 | 0.124 | 54 / 1577 | 204 (80.4%) | 10 | 132 | pass |
| **E3** | **0.413** | 0.163 | 57 / 1577 | 234 (69.7%) | 34 | 132 | **pass, selected** |
| E4 | 0.383 | 0.083 | 49 / 1577 | 222 (80.6%) | 19 | 133 | pass |

All five models were re-evaluated back to back on the same machine (Apple M1,
8 GB) so the latency column is comparable. Each fine-tuned model's predictions
were then recomputed from its weights with no cached scores, and the resulting
metrics were identical to the committed ones.

## Decision

**The pre-registered rule selects E3.** All four candidates pass every gate
(macro-F1 gain >= 0.05, false-empty events <= 180, CPU latency <= 1.5x); the
rule then takes the highest macro-F1.

The rule was applied as written. What it does not settle:

- **E3's lead is small and not robust.** It beats E2 by 0.010 macro-F1 on four
  cameras, with no interval on macro-F1. On the policy-validation cameras alone
  E1 is highest (E1 0.384, E3 0.382, E2 0.378); E3's advantage comes from the
  calibration cameras (0.452 vs E2 0.444).
- **E3 is worse on two product measures the rule does not include.** Balanced
  sampling makes the model predict a species more often: at the reference
  thresholds its accepted labels are 69.7% correct (E2: 80.4%), and it accepts
  34 unsupported-animal events as a known species (E2: 10). Calibration and
  threshold choice decide the final operating point, so these are reported,
  not used to override the rule after the fact.
- **Where E3 is better:** the best night-time macro-F1 (0.356; E2 0.331) and
  the fewest night false-empty events (19 of 1028; baseline 121), and higher
  recall for rabbit (0.43), raccoon (0.39), and opossum (0.44).

A future selection rule should add accepted-label precision and unsupported
acceptance as gates; changing this one now would be choosing the rule after
seeing the results.

## Failure analysis

- **Camera 51 dominates missed animals.** Every fine-tuned model still filters
  43-51 of its 396 animal events as empty, while the other three cameras drop
  to 0-7. Many of these are squirrels (unsupported) and small rabbits and
  opossums with frame P(empty) above 0.95. New cameras need per-camera audits
  before aggressive filtering.
- **Cat recall is poor for every model** (0.08-0.21); cats are confused with
  dog, raccoon, and coyote.
- **Unsupported animals are accepted confidently.** Squirrels and rodents are
  labeled rabbit, coyote, or dog at confidence up to 1.00. Confidence alone
  does not reject unfamiliar inputs; the distance-based unfamiliar-input score
  is still to be evaluated.
- **Seen vs unseen gap.** Macro-F1 on held-out sequences from training cameras
  is 0.75-0.76 for every fine-tuned model, against 0.38-0.41 on unseen cameras.
  Random-sequence evaluation would overstate performance by about 0.35.
- **No product target is met yet.** At the reference thresholds no model
  retains 98% of animal events (E3: 96.4%; 98.9% at P(empty) >= 0.99), and
  accepted precision reaches 95% only for E1 at species >= 0.99, on 28
  events (E3 at 0.99: 51 events, 88.2%). CCT20 development events are only ~15%
  empty, so empty filtering alone cannot cut review by half; review savings
  have to come from confident species labels.

## Caveats

- Training times (E1 17 min, E2 41, E3 70, E4 45) are wall-clock on a shared
  8 GB laptop under different memory pressure and are not comparable between
  experiments.
- E2 was trained from a working tree with uncommitted changes (recorded as
  `dirty` in its lineage); its code commit is `3e011a4`.
- Results rest on four development cameras.

## Next

Calibrate E3 on the calibration partition, then choose the empty-filter and
species-acceptance thresholds on policy validation, before the final test is
opened.
