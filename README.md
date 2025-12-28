# 🔐 JAILS: Jailbreak Instruction Leakage Detection System

**JAILS** is a hybrid detection system designed to identify **jailbreak and prompt-injection attempts** against Large Language Models (LLMs).
It combines **semantic similarity, linguistic analysis, pattern-based heuristics, and classical machine-learning models** to detect adversarial prompts beyond simple keyword matching.

This project focuses on **prompt-level security analysis** and is intended as a **research and engineering prototype** for LLM safety exploration.

---

## 🎯 Problem Overview

LLMs are vulnerable to a wide range of prompt-based attacks, including:

- Instruction overrides (e.g., *“ignore all previous instructions”*)
- Role-play jailbreaks (e.g., *“act as an unrestricted AI”*)
- Paraphrased attacks that evade keyword filters
- Context flooding through repeated adversarial instructions
- Privilege escalation or authority-based manipulation

Rule-based filters alone fail to generalize.  
**JAILS** addresses this by aggregating multiple weak signals into a stronger detection decision.

---

## 🏗️ System Architecture

```
Input Prompt (+ optional context)
        ↓
Feature Extraction Layer
- Linguistic & structural features
- Pattern-based manipulation signals
- Statistical text representations
        ↓
Detection Engine
- ML classifier (Gradient Boosting / RF / LR)
- Anomaly detection (LOF)
- Rule-based fallback
        ↓
Output
SAFE / JAILBREAK
+ confidence & risk score
```

---

## ✨ Key Capabilities

### 🔍 Multi-Layer Detection
- Semantic similarity against known jailbreak patterns
- Linguistic features: repetition, length, structure, readability cues
- Pattern matching: instruction overrides, role-play triggers, coercion
- Statistical signals: TF-IDF similarity, clustering, anomaly detection

### 🎛️ Machine-Learning Pipeline
- Feature scaling and preprocessing
- Supervised classifiers:
  - Gradient Boosting (primary)
  - Random Forest / Logistic Regression (baseline comparisons)
- Local Outlier Factor (LOF) for unseen / zero-day attacks
- Heuristic fallback when confidence is low

---

## 🌍 Attack Coverage (Examples)

| Category | Example | Detection Signal |
|--------|--------|------------------|
| Instruction Override | "Ignore previous rules" | Regex + semantics |
| Role-Play Jailbreak | "Act as DAN" | Patterns + structure |
| Context Flooding | Repeated adversarial text | Repetition + anomaly |
| Privilege Escalation | "Admin override enabled" | Patterns |
| Disguised Intent | "For academic research only…" | Linguistic + semantic |

---

## 📂 Repository Structure

```
jails/
├── flooding2.py          # Main detection engine & training logic
├── multi10.py            # Feature extraction & pattern analysis
├── classifier/
│   └── models/           # Trained model artifacts (joblib)
├── README.md
└── requirements.txt
```

---

## 🚀 Quick Start

### 1️⃣ Installation
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the detector
```bash
python flooding2.py
```

The script demonstrates:
- Model training
- Evaluation metrics
- Sample prompt classification

---

## 📈 Model Outputs

For each prompt, JAILS produces:
- Classification: `SAFE` or `JAILBREAK`
- Confidence score
- Risk / threat level
- Feature-based reasoning signals

This makes the system **interpretable**, not a black box.

---

## 🧠 What This Project Demonstrates

- Understanding of LLM jailbreak & prompt-injection risks
- Hybrid detection (rules + ML + statistics)
- Feature engineering for NLP security
- Anomaly detection for unseen attacks
- Research-style experimentation with classifiers

---

## ⚠️ Disclaimer

This project is an **experimental prototype** intended for **learning and research purposes**.
It is **not a production-ready security system**.

---

## 📄 License

MIT License
