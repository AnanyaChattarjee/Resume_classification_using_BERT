# 📄 Resume Job Role Classification using BERT (Transformer)

This project is a **Resume-to-Job Role Classification System** built using a **custom BERT-style Transformer Encoder**.  
It takes resume text as input and predicts the most suitable **job category/role**.

The pipeline includes **PDF resume text extraction**, preprocessing, tokenization, padding, label encoding, and a Transformer-based classification model.

---

## 🚀 Features
- Extracts text from resume PDFs using `pdfplumber`
- Tokenizes resumes using `bert-base-uncased` tokenizer
- Custom implementation of:
  - Self-Attention (Q, K, V)
  - Multi-Head Attention
  - Encoder Layers
  - Positional Embeddings
  - `[CLS]` token based classification
- Supports **partial fine-tuning** using pretrained HuggingFace BERT embeddings
- Predicts job role from resume text

---

## 🧠 Model Architecture
The model follows the Transformer Encoder design:\

Input Token IDs\
↓\
Token Embedding + Positional Embedding\
↓\
Encoder Layer Stack (Multi-Head Attention + FFN)\
↓\
[CLS] Token Representation\
↓\
Classification Head\
↓\
Job Role Prediction\


---


## 📂 Dataset Structure
Dataset is organized as folders, where each folder name represents the job role:
``` bash
data/
├── ENGINEERING/
├── FITNESS/
├── ARTS/
├── SALES/
└── ...
```

Each folder contains multiple resume PDF files.

---

## ⚙️ Installation

```bash
pip install pdfplumber tiktoken tqdm transformers torch pandas numpy
```

---

## 📌 Tech Stack
- Python  
- PyTorch  
- Transformers (HuggingFace)  
- PDFPlumber  
- NumPy, Pandas  

---

## 📈 Future Improvements
- Add sliding window chunking for resumes longer than 512 tokens  
- Implement attention masking for PAD tokens  
- Compare results with HuggingFace `BertForSequenceClassification`  
- Deploy the model using Streamlit  

---

## 👩‍💻 Author
**Ananya Chattarjee**  
📍 Jaipur, Rajasthan, India  
🔗 GitHub: [AnanyaChattarjee](https://github.com/AnanyaChattarjee)  
