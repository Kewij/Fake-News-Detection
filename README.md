# Fake-News-Detection

## Overview
This project focuses on building an efficient **Fake News Detector** by leveraging **Pretrained Transformer Models** (specifically BERT). The objective is to classify news articles as *real* or *fake* with high accuracy.

The workflow includes:
- Data collection and preprocessing
- Fine-tuning a pretrained BERT model
- Comparing results between a frozen BERT model and a fine-tuned BERT model
- Building an inference pipeline for real-world usage

---

## Dataset
Two datasets were utilized during the project:
1. **Premier Dataset** — Initially used but found too simple.
2. **Second Dataset** — Adopted for a more realistic and challenging classification task.

The latest versions of the datasets were downloaded and used for training and evaluation.

---

## Methodology
- **Pretrained BERT** was first used in a frozen state, where its internal weights were not updated during training.
- In a second approach, BERT was **fine-tuned** to adapt better to the specific fake news detection task.
- Optimization was performed using the **Adam optimizer** with a learning rate of `1e-5`.
- Model validation was conducted throughout the training process to monitor performance.

---

## Results
The project clearly demonstrated the benefits of fine-tuning:

| Method          | Accuracy | F1 Score |
| --------------- | -------- | -------- |
| Frozen BERT     | 94.8%    | 0.947    |
| Fine-tuned BERT | 99.97%   | 0.9998   |

Fine-tuning significantly enhanced the model’s performance, achieving near-perfect scores.

---

## Conclusion
This project highlights the effectiveness of **transfer learning** and **fine-tuning** in tackling text classification problems like fake news detection. Fine-tuning a pretrained model leads to superior results compared to using a frozen model.

---

## Future Work
- Experiment with other transformer-based models like **RoBERTa**, **DeBERTa**, or **DistilBERT**.
- Evaluate performance on noisier, real-world datasets.
- Optimize the model for faster inference and potential deployment on mobile or web applications.

---
