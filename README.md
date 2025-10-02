# LoRA from Scratch: Fine-Tuning BERT on SST-2 (with PyTorch)

This project implements **Low-Rank Adaptation (LoRA)** manually in PyTorch, injecting it into a BERT model for sentiment classification on the **SST-2 dataset** (GLUE benchmark). It demonstrates parameter-efficient fine-tuning using only ~0.5% of BERT’s weights.

> 🔧 Built from scratch — no PEFT, no adapters — just pure PyTorch and Transformers.

---

## ⚙️ Try It Yourself

[Open in Colab](https://colab.research.google.com/drive/1wpDmCpdYWE4sy12AD69ftBsSmAkj7BI6?usp=sharing)  
Clone this repo or open the notebook in Colab to reproduce results.

---

## 🚀 What This Project Demonstrates

- ✅ Manual implementation of the LoRA mechanism (`A @ B` low-rank updates)
- ✅ Patching `BertSelfAttention` layers with LoRA adapters
- ✅ Freezing base BERT weights, updating only LoRA params
- ✅ Training on a real-world NLP task: SST-2
- ✅ Inference on validation set after fine-tuning

---

## 🧪 Dataset

- **SST-2** (Stanford Sentiment Treebank, binary classification)
- Loaded via `datasets` library
- Split into 90% train / 10% validation
- Tokenized using `BertTokenizerFast`

---

## 📦 LoRA Architecture

- Injected into `query` and `value` layers of BERT attention
- LoRA rank `r = 4`, scaling factor `alpha = 16`
- `LoRALinear` module wraps the original `nn.Linear`
- Final BERT → classifier head → `CrossEntropyLoss`

---

## 📊 Results

| Metric              | Value        |
|---------------------|--------------|
| Trainable Params    | ~500K (vs 110M full) |
| Validation Accuracy | ~89–91%      |
| Epochs              | 3            |
| Inference Supported | ✅ Yes       |

---

## 🛠️ Tech Stack

- `transformers` (BERT backbone)
- `datasets` (SST-2)
- `torch` (LoRA + training)
- `sklearn` (metrics)

---

## 💡 Why This Project?

> Fine-tuning massive language models is expensive. LoRA lets us do it efficiently.

By updating only low-rank matrices and freezing most of BERT, we:
- Save compute & memory
- Reduce overfitting risk
- Enable deployment in low-resource settings

---

## 📬 Author

**Arjun Jagdale**  
Machine Learning • AI • Open Source Contributor  
📍 Pune, India  
📧 arjunjagdale.sits.entc@email.com
