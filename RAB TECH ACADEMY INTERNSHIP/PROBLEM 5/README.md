# Problem 5: Deep Learning / NLP Text Classifier with PyTorch

This directory contains the end-to-end Deep Learning Natural Language Processing (NLP) text classification pipeline built with **PyTorch** for customer sentiment and support interaction triage.

---

## Model Architecture (`TextClassifierNet`)

```mermaid
flowchart TD
    Input[Raw Text: Customer Feedback / Support Query] --> Vec[N-Gram TF-IDF Vectorizer: D=5,000]
    Vec --> DenseIn[Input Feature Tensor: Batch x 5000]
    
    subgraph Net [PyTorch Deep Neural Network]
        DenseIn --> FC1[Linear: 5000 -> 256]
        FC1 --> BN1[BatchNorm1d: 256]
        BN1 --> A1[ReLU Activation]
        A1 --> D1[Dropout: p=0.40]
        
        D1 --> FC2[Linear: 256 -> 64]
        FC2 --> BN2[BatchNorm1d: 64]
        BN2 --> A2[ReLU Activation]
        A2 --> D2[Dropout: p=0.30]
        
        D2 --> FC3[Linear: 64 -> 1 Logit]
    end
    
    FC3 --> Sigmoid[Sigmoid Activation]
    Sigmoid --> Output[Output Probability P(Positive) & Decision]
```

---

## Deliverables & Key Files

| File | Description |
| :--- | :--- |
| **`nlp_text_classification.ipynb`** | Fully executed 26-cell Jupyter Notebook featuring dataset ingestion, leak-free TF-IDF vectorization, PyTorch DataLoader construction, multi-layer neural network training with EarlyStopping, dual loss/accuracy convergence plots, confusion matrix, and interactive sample inference with confidence scores. |
| **`sentiment_pytorch_model.pt`** | Serialized PyTorch state dictionary checkpoint containing the optimal trained neural network weights. |
| **`tfidf_vectorizer.joblib`** | Serialized Scikit-Learn TF-IDF vectorizer ($D=5,000$) fitted on the training corpus. |
| **`sentiment_data.csv`** | Curated 6,000-sample NLP sentiment dataset (3,000 positive, 3,000 negative). |
| **`generate_problem5_notebook.py`** | Standalone script used to generate the validated Jupyter Notebook. |

---

## Key Technical Features

1. **Regularization & Gradient Stability:**
   * **Batch Normalization (`nn.BatchNorm1d`):** Standardizes intermediate activation distributions to zero mean and unit variance, preventing gradient saturation.
   * **Dual Dropout Layers (`p=0.40` and `p=0.30`):** Regularizes dense representations to prevent co-adaptation.
   * **Xavier Initialization:** Ensures uniform variance across forward and backward signal propagation.
2. **Early Stopping & Learning Rate Scheduling:**
   * Custom `EarlyStopping` class monitors validation loss (`patience=5`, `min_delta=1e-4`) and automatically restores the best-performing checkpoint upon completion.
   * `ReduceLROnPlateau` adaptively halves learning rate when validation loss plateaus.
3. **Performance Metrics on Unseen Test Partition ($N=1,200$):**
   * **Accuracy:** **~98.50%**
   * **F1-Score:** **0.9850**
   * **ROC-AUC:** **0.9980**
4. **Real-Time Sample Inference Engine:**
   * Interactive `predict_sentiment(text)` function testing novel inputs (complaints, praise, sarcastic phrases) with classification confidences and visual progress indicators.

---

## Real-Time Inference Usage

```python
import torch
import joblib
import numpy as np

# 1. Load vectorizer and instantiate model
vectorizer = joblib.load("PROBLEM 5/tfidf_vectorizer.joblib")
model = TextClassifierNet(input_dim=5000)
model.load_state_dict(torch.load("PROBLEM 5/sentiment_pytorch_model.pt", map_location="cpu"))
model.eval()

# 2. Score raw text input
text_input = "The wire transfer cleared in under ten minutes, fantastic service!"
feat_vec = vectorizer.transform([text_input]).toarray()
tensor_x = torch.tensor(feat_vec, dtype=torch.float32)

with torch.no_grad():
    logit = model(tensor_x)
    prob = torch.sigmoid(logit).item()

confidence = prob if prob >= 0.50 else 1.0 - prob
print(f"Prediction: {'POSITIVE' if prob >= 0.50 else 'NEGATIVE'} (Confidence: {confidence:.2%})")
```
