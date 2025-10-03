## ⚙️ Try It Yourself

[Open in Colab](https://colab.research.google.com/drive/1wpDmCpdYWE4sy12AD69ftBsSmAkj7BI6?usp=sharing)  

# LoRA Fine-Tuning for BERT on SST-2 Sentiment Analysis

A comprehensive implementation of Low-Rank Adaptation (LoRA) for efficient fine-tuning of BERT models on the Stanford Sentiment Treebank (SST-2) dataset for binary sentiment classification.

## Overview

This notebook demonstrates how to implement and apply LoRA to BERT for sentiment analysis, achieving competitive performance while training only a small fraction of the model's parameters. LoRA enables efficient fine-tuning by injecting trainable low-rank matrices into the attention layers while keeping the original pre-trained weights frozen.

## Key Features

- **Parameter Efficiency**: Trains only 147,456 parameters (~0.13% of BERT-base) using LoRA adapters
- **Custom LoRA Implementation**: Built-from-scratch LoRA module with configurable rank and alpha
- **High Performance**: Achieves ~91.7% accuracy on SST-2 validation set
- **Attention Layer Injection**: Applies LoRA to Query and Value projections in self-attention
- **Real-time Inference**: Includes sentiment prediction pipeline for custom text

## Dataset

**SST-2 (Stanford Sentiment Treebank - Binary)**
- **Task**: Binary sentiment classification (Positive/Negative)
- **Training samples**: 67,349
- **Validation samples**: 872
- **Test samples**: 1,821
- **Source**: GLUE benchmark via Hugging Face datasets

## Architecture

### LoRA Configuration

```python
r (rank) = 4          # Low-rank dimension
alpha = 16            # Scaling factor
scaling = alpha / r   # Final scaling: 4.0
```

### Modified Layers

LoRA adapters are injected into:
- **Query projections** in all 12 BERT attention layers
- **Value projections** in all 12 BERT attention layers

The forward pass computes:
```
output = W(x) + (alpha/r) * B(A(x))
```

Where:
- W: Original frozen weight matrix (768 × 768)
- A: Down-projection matrix (4 × 768) - trainable
- B: Up-projection matrix (768 × 4) - trainable

## Requirements

```bash
transformers
datasets
scikit-learn
torch
numpy
tqdm
```

Install dependencies:
```bash
pip install transformers datasets scikit-learn --quiet
```

## Hardware Requirements

- **Recommended**: GPU with CUDA support (T4, V100, A100, etc.)
- **Minimum**: CPU (training will be significantly slower)
- **RAM**: 8GB+ recommended
- **VRAM**: 4GB+ for GPU training

## Usage

### 1. Setup and Training

Run all cells sequentially in Google Colab or Jupyter:

```python
# The notebook will automatically:
# 1. Load and tokenize SST-2 dataset
# 2. Initialize BERT-base-uncased
# 3. Inject LoRA adapters into attention layers
# 4. Freeze all parameters except LoRA matrices
# 5. Train for 3 epochs with AdamW optimizer
```

### 2. Training Configuration

```python
batch_size = 32           # Training batch size
val_batch_size = 64       # Validation batch size
learning_rate = 1e-4      # AdamW learning rate
epochs = 3                # Training epochs
max_length = 128          # Maximum sequence length
```

### 3. Inference on Custom Text

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def predict_sentiment(text, model, tokenizer):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", 
                      padding=True, truncation=True, 
                      max_length=128).to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred].item()
    
    label = "Positive" if pred == 1 else "Negative"
    return label, confidence

# Example usage
text = "This movie was absolutely fantastic!"
label, confidence = predict_sentiment(text, model_lora, tokenizer)
print(f"{label} ({confidence:.2f})")
```

## Results

### Performance Metrics

| Metric | LoRA BERT |
|--------|-----------|
| **Validation Accuracy** | 91.74% |
| **Training Loss (Epoch 3)** | 0.2150 |
| **Trainable Parameters** | 147,456 |
| **Total Parameters** | ~110M |
| **% Trainable** | 0.13% |

### Classification Report

```
                 precision    recall  f1-score   support
Negative (0)        0.897     0.939     0.918       428
Positive (1)        0.939     0.896     0.917       444

accuracy                                0.917       872
macro avg           0.918     0.918     0.917       872
weighted avg        0.918     0.917     0.917       872
```

### Training Progress

```
Epoch 1/3: Train Loss: 0.3326 | Val Accuracy: 0.8991
Epoch 2/3: Train Loss: 0.2424 | Val Accuracy: 0.9094
Epoch 3/3: Train Loss: 0.2150 | Val Accuracy: 0.9174
```

## Implementation Details

### LoRA Module Architecture

```python
class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=4, alpha=16):
        # Down-projection: (batch, seq, 768) → (batch, seq, 4)
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        
        # Up-projection: (batch, seq, 4) → (batch, seq, 768)
        self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)
        
        # Scaling factor
        self.scaling = alpha / r
```

### Parameter Freezing Strategy

1. All BERT pre-trained weights are frozen
2. Only LoRA matrices (A and B) are trainable
3. Classifier head remains trainable (default BERT behavior)
4. Total trainable: 147,456 parameters

## Advantages of LoRA

1. **Memory Efficient**: Only stores small adapter matrices
2. **Fast Training**: Fewer parameters to update
3. **Modularity**: Easy to swap adapters for different tasks
4. **No Inference Overhead**: Can merge adapters with base weights
5. **Storage Efficient**: Save only 0.5MB vs 440MB for full model

## Potential Improvements

- Experiment with different rank values (r=8, r=16)
- Apply LoRA to all linear layers (FFN, Key projections)
- Implement learning rate scheduling
- Add gradient accumulation for larger effective batch sizes
- Try different alpha values for scaling
- Experiment with LoRA dropout
- Test on other GLUE tasks

## Citation

If you use this implementation, please cite:

```bibtex
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size to 16 or 8
- Reduce max_length to 64
- Use gradient accumulation

### Low Accuracy
- Increase training epochs (5-10)
- Try different learning rates (1e-3, 5e-5)
- Increase LoRA rank to 8 or 16

### Slow Training
- Enable GPU in Colab: Runtime → Change runtime type → GPU
- Use mixed precision training with `torch.cuda.amp`
- Increase batch size if memory allows

## License

This implementation is provided for educational purposes. BERT and the SST-2 dataset have their own respective licenses.

## Acknowledgments

- **BERT**: Devlin et al., Google AI Language
- **LoRA**: Hu et al., Microsoft
- **SST-2**: Socher et al., Stanford NLP
- **Hugging Face**: Transformers and Datasets libraries
