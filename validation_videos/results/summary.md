# Validation Results — Per-Clip Analysis

- Total clips: **8**
- Accuracy: **50.0%** (4/8)
- Precision: **0.000**  ·  Recall: **0.000**  ·  F1: **0.000**

## Confusion Matrix

|                  | pred=fatigue | pred=non-fatigue |
|------------------|:------------:|:----------------:|
| **gt=fatigue**     | 0 | 4 |
| **gt=non-fatigue** | 0 | 4 |

## Per-Clip Details

| ID | GT | Predicted | OK | MaxScore | MeanScore | Yawn | Microsleep | Nod | Face% |
|----|------|------|:---:|--------:|---------:|----:|----:|----:|------:|
| 23 | non-fatigue | non-fatigue | OK | 0.525 | 0.247 | 0 | 0 | 0 | 99.0% |
| 24 | fatigue | non-fatigue | X | 0.567 | 0.381 | 0 | 0 | 0 | 100.0% |
| 25 | fatigue | non-fatigue | X | 0.592 | 0.356 | 0 | 0 | 2 | 100.0% |
| 26 | non-fatigue | non-fatigue | OK | 0.547 | 0.221 | 0 | 0 | 0 | 100.0% |
| 27 | non-fatigue | non-fatigue | OK | 0.492 | 0.235 | 0 | 0 | 0 | 100.0% |
| 28 | fatigue | non-fatigue | X | 0.535 | 0.343 | 0 | 0 | 0 | 100.0% |
| 29 | non-fatigue | non-fatigue | OK | 0.639 | 0.240 | 0 | 0 | 1 | 99.7% |
| 30 | fatigue | non-fatigue | X | 0.507 | 0.253 | 0 | 0 | 0 | 99.7% |