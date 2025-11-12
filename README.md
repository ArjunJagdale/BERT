# LoRA Fine-tuning for BERT on SST-2 Sentiment Analysis

## Overview

This project implements **Low-Rank Adaptation (LoRA)** for efficient fine-tuning of BERT on the Stanford Sentiment Treebank (SST-2) dataset. LoRA enables parameter-efficient transfer learning by injecting trainable low-rank matrices into the attention layers while keeping the pretrained weights frozen.

## Key Results

### Model Performance
- **Final Validation Accuracy**: **91.74%**
- **Training Epochs**: 3
- **Batch Size**: 32 (training), 64 (validation)

### Training Progress
| Epoch | Train Loss | Validation Accuracy |
|-------|------------|---------------------|
| 1     | 0.3326     | 89.91%             |
| 2     | 0.2424     | 90.94%             |
| 3     | 0.2150     | 91.74%             |

### Classification Metrics (Epoch 3)
| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Negative | 0.897     | 0.939  | 0.918    | 428     |
| Positive | 0.939     | 0.896  | 0.917    | 444     |
| **Weighted Avg** | **0.918** | **0.917** | **0.917** | **872** |

### Parameter Efficiency
- **LoRA Trainable Parameters**: **147,456** (~0.13% of BERT-base's 110M parameters)
- **LoRA Configuration**:
  - Rank (r): 4
  - Alpha (α): 16
  - Scaling factor: α/r = 4

This represents a **~750x reduction** in trainable parameters compared to full fine-tuning, while maintaining competitive performance.

## Architecture Details

### LoRA Implementation
LoRA modifies the attention mechanism by adding low-rank decomposition matrices:

```
W' = W + (α/r) · B · A
```

Where:
- `W`: Original frozen weight matrix
- `A`: Down-projection matrix (r × d_in)
- `B`: Up-projection matrix (d_out × r)
- `r`: Rank (dimensionality of the bottleneck)
- `α`: Scaling hyperparameter

Code example:
```
class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 4, alpha: int = 16):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.alpha = alpha

        # Original frozen weight
        self.weight = original_linear.weight
        self.bias = original_linear.bias

        # LoRA adapters (A: down-projection, B: up-projection)
        self.A = nn.Parameter(torch.randn(r, self.in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(self.out_features, r) * 0.01)

        # Scaling factor
        self.scaling = self.alpha / self.r

        # Freeze the original weight
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x):
        # LoRA: W(x) + alpha/r * BA(x)
        lora_update = (x @ self.A.T) @ self.B.T
        return nn.functional.linear(x, self.weight) + self.scaling * lora_update

```

### Injection Points
LoRA adapters are injected into:
- Query projection layers (`attention.self.query`)
- Value projection layers (`attention.self.value`)

All 12 BERT encoder layers receive LoRA adapters (24 total injection points).

## Dataset

**SST-2 (Stanford Sentiment Treebank v2)**
- **Task**: Binary sentiment classification
- **Training samples**: 67,349
- **Validation samples**: 872
- **Test samples**: 1,821
- **Classes**: Negative (0), Positive (1)
- **Max sequence length**: 128 tokens

## Requirements

```bash
pip install transformers datasets scikit-learn torch
```

### Dependencies
- `transformers`: Hugging Face Transformers library
- `datasets`: Hugging Face Datasets library
- `scikit-learn`: Evaluation metrics
- `torch`: PyTorch framework
- `tqdm`: Progress bars

## Usage

### 1. Environment Setup
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 2. Load Dataset and Tokenizer
```python
from datasets import load_dataset
from transformers import BertTokenizer

dataset = load_dataset("glue", "sst2")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
```

### 3. Inject LoRA into BERT
```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2
)
inject_lora_into_bert(model, r=4, alpha=16)
model.to(device)
```

### 4. Freeze Base Model, Enable LoRA Parameters
```python
for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "A" in name or "B" in name:
        param.requires_grad = True
```

### 5. Training
```python
from torch.optim import AdamW

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-4
)

# Train for 3 epochs (see notebook for full training loop)
```

### 6. Inference
```python
def predict_sentiment(text, model, tokenizer):
    model.eval()
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=128
    ).to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred].item()
    
    label = "Positive" if pred == 1 else "Negative"
    return label, confidence
```

## Sample Predictions

| Sentence | Prediction | Confidence |
|----------|------------|------------|
| "The movie was fantastic and thrilling!" | Positive | 0.98 |
| "I wouldn't recommend it to anyone." | Negative | 0.90 |
| "It was okay, not great but not bad." | Positive | 0.93 |
| "This is one of the best performances I've seen." | Positive | 0.98 |
| "The film lacked a solid storyline." | Negative | 0.98 |

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Learning Rate | 1e-4 |
| Optimizer | AdamW |
| Batch Size (train) | 32 |
| Batch Size (val) | 64 |
| Epochs | 3 |
| Max Sequence Length | 128 |
| LoRA Rank (r) | 4 |
| LoRA Alpha (α) | 16 |

## Advantages of LoRA

1. **Memory Efficiency**: Only 147K trainable parameters vs. 110M for full fine-tuning
2. **Training Speed**: Faster convergence due to fewer parameters to update
3. **Storage**: Multiple task-specific adapters can be stored without duplicating the base model
4. **Performance**: Achieves 91.74% accuracy, comparable to full fine-tuning
5. **Modularity**: Easy to switch between different task adapters at inference time

## File Structure

```
.
├── LoRA_BERT_SST2_CLEAN.ipynb    # Main notebook with implementation
└── README.md                      # This file
```

## References

- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **BERT**: [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- **SST-2 Dataset**: [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/)

## License

This project is for educational purposes. Please refer to the respective licenses of BERT, Hugging Face libraries, and the SST-2 dataset.

## Citation

If you use this implementation, please cite the original LoRA paper:

```bibtex
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```

---

**Note**: This implementation demonstrates LoRA on BERT-base for sentiment analysis. The technique is applicable to other transformer models and tasks with minimal modifications.
