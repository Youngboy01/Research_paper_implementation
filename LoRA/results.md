## Results

### Experiment Setup

We compared a standard Multi-Layer Perceptron (MLP) against a LoRA-adapted MLP on the same dataset for 5 training epochs using CPU execution.

### Performance Comparison

| Metric               | Standard MLP | LoRA MLP   |
| -------------------- | ------------ | ---------- |
| Trainable Parameters | 407,050      | 3,636      |
| Parameter Reduction  | -            | 99.11%     |
| Final Training Loss  | 0.0310       | 0.5511     |
| Test Accuracy        | **97.36%**   | **84.65%** |
| Training Time        | 123.02 s     | 203.08 s   |

### Training Progress

#### Standard MLP

| Epoch | Loss   |
| ----- | ------ |
| 1     | 0.2047 |
| 2     | 0.0857 |
| 3     | 0.0555 |
| 4     | 0.0419 |
| 5     | 0.0310 |

#### LoRA MLP

| Epoch | Loss   |
| ----- | ------ |
| 1     | 1.1077 |
| 2     | 0.7253 |
| 3     | 0.6374 |
| 4     | 0.5849 |
| 5     | 0.5511 |

### Analysis

The standard MLP achieved a test accuracy of **97.36%** with **407,050 trainable parameters**. In contrast, the LoRA-based MLP reduced the number of trainable parameters to only **3,636**, representing a **99.11% reduction** in trainable parameters.

This substantial parameter reduction came at the cost of predictive performance, with the LoRA model achieving **84.65% test accuracy**, approximately **12.7 percentage points lower** than the standard MLP. The LoRA model also converged more slowly, maintaining a higher training loss throughout the experiment.

These results demonstrate the trade-off between parameter efficiency and model performance. While LoRA dramatically decreases the number of trainable parameters, the reduced adaptation capacity leads to lower accuracy on this task.
