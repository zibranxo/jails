# 🔐 JAILS: Jailbreak & Instruction Leakage Detection System

**JAILS** is a research-oriented **jailbreak / prompt-injection detection system** designed to identify unsafe or adversarial user prompts intended to bypass Large Language Model (LLM) safety mechanisms.

This project was developed as part of an **AIMS-DTU internship / research exploration** and focuses on combining **semantic similarity, linguistic analysis, pattern-based heuristics, and machine learning** for robust detection.

---

## 🎯 Problem Statement

Large Language Models are vulnerable to **jailbreak and prompt-injection attacks**, such as:

- Instruction overrides (e.g., “ignore previous rules”)
- Role-play based safety bypasses
- Paraphrased or obfuscated attacks that evade keyword filters

Simple rule-based systems fail to generalize.  
JAILS explores a **hybrid detection approach** that goes beyond keywords.

---

## 🧠 High-Level Approach

JAILS combines multiple detection signals:

1. **Semantic similarity** to known jailbreak prompts  
2. **Linguistic and structural features**  
3. **Pattern-based heuristics**  
4. **Supervised machine learning**  
5. **Anomaly detection**  

The system is model-agnostic and can be deployed as a **pre-filter or guard layer** for LLMs.

---

## 🧩 Core Components

### 1️⃣ Semantic Embedding Manager
- Encodes prompts using transformer-based embeddings  
- Compares similarity against known **jailbreak** and **safe** prompts  
- Enables detection of paraphrased attacks  

---

### 2️⃣ Advanced Feature Extractor

Extracted feature groups include:

- **Semantic features**: jailbreak similarity, safe similarity, confidence gap  
- **Linguistic features**: word count, sentence count, readability cues  
- **Sentiment features**: polarity scores  
- **Pattern features**: instruction overrides, manipulation phrases  

---

### 3️⃣ Jailbreak Detection Engine

The final decision engine:
- Aggregates extracted features  
- Uses a **Gradient Boosting classifier**  
- Applies **Local Outlier Factor (LOF)** for anomaly detection  
- Falls back to heuristic rules when needed  

Each prompt is classified as:
- `SAFE`
- `JAILBREAK`

Along with confidence and threat scores.

---

## 🧪 Training & Evaluation

Training samples follow the format:

```python
(prompt_text, task_intent, label)
```

Where:
- `label = 1` → jailbreak  
- `label = 0` → safe  

The training pipeline includes:
- Feature scaling
- Classifier training
- Anomaly model fitting
- Validation metrics (accuracy, AUC)

---

## 📂 Repository Structure

```
├── flooding2.py        # Main detection system
├── README.md           # Documentation
└── requirements.txt    # Dependencies
```

---

## 🚀 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the detector
```bash
python flooding2.py
```

The script includes example training data and test prompts.

---

## 🧠 What This Project Demonstrates

- Understanding of **LLM safety vulnerabilities**
- Hybrid detection (rules + ML + semantics)
- Feature engineering for NLP security
- Generalization beyond keyword matching
- Research-oriented system design

---

## 🔮 Future Work

- Jailbreak taxonomy classification  
- FAISS-based semantic retrieval  
- Continual / online learning  
- Integration with live LLM APIs  

---

## 📄 Disclaimer

This project is intended for **research and educational use only**.  
It is **not a production-ready security system**.

---

## 👤 Author

Developed by **Zibran**  
Research focus: **LLM Safety & Alignment**
