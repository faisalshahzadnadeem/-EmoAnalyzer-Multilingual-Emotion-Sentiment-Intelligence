# 🎭 EmoAnalyzer — Multilingual Emotion & Sentiment Intelligence

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red)
![Cohere](https://img.shields.io/badge/Cohere-API-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

**EmoAnalyzer** is an advanced AI-powered multilingual emotion detection system designed to identify **8 core human emotions** from text across **100+ languages**.
Unlike traditional sentiment tools that classify text as only positive or negative, EmoAnalyzer provides **deep emotional insights**, probability scoring, and interactive visual analytics.

This tool is ideal for researchers, businesses, developers, and mental health applications requiring **nuanced emotional intelligence**.

---

## ✨ Key Features

✔ **8 Emotion Categories**
Anger • Anticipation • Disgust • Fear • Joy • Sadness • Surprise • Trust

✔ **Multilingual Support**
Works across 100+ languages using Cohere multilingual embeddings.

✔ **Real-time Analysis**
Fast inference with progress tracking.

✔ **Interactive Visualizations**
Includes radar charts, probability distributions, and comparative graphs.

✔ **Batch Processing**
Analyze thousands of texts in a single run.

✔ **History Tracking**
Save and review previous results.

✔ **Mental Wellness Suggestions**
Provides coping and emotional guidance based on detected mood.

✔ **Export Functionality**
Download results in CSV format.

---

## 🚀 Quick Start

### ✅ Prerequisites

* Python **3.8 or above**
* Cohere API Key
  Get one here → https://dashboard.cohere.ai/register

---

### ⚙ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/emoanalyzer.git
cd emoanalyzer

# Create virtual environment
python -m venv venv

# Activate environment
# Linux / Mac
source venv/bin/activate
# Windows
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set Cohere API Key
# Linux / Mac
export COHERE_API_KEY="your-api-key"

# Windows
# set COHERE_API_KEY="your-api-key"

# Run application
streamlit run sentiment.py
```

---

## 📂 Project Structure

```
emoanalyzer/
│
├── sentiment.py               # Main Streamlit application
├── utils.py                   # Core helper functions
├── requirements.txt           # Python dependencies
│
├── data/
│   ├── emotions/              # Emotion GIFs and assets
│   ├── models/                # Trained ML models
│   └── xed_with_embeddings.json
│
└── README.md                  # Documentation
```

---

## 🎯 Usage

### 🔹 Single Text Analysis

Input a sentence through the UI:

```python
"I'm feeling absolutely wonderful today!"
```

Example Output:

```
Joy: 92%
Anticipation: 45%
Trust: 30%
```

---

### 🔹 Batch Processing

* Upload multiple texts (one per line).
* Supports up to **2048 texts per batch**.
* Download results as CSV for further analysis.

---

## 🧠 System Architecture

```
User Input
   ↓
Cohere Multilingual Embeddings
   ↓
Emotion Classification Model
   ↓
Probability Scoring
   ↓
Visualization & Insights
```

---

## 📊 Performance Metrics

| Metric              | Score        |
| ------------------- | ------------ |
| Accuracy            | 87.5%        |
| Avg Response Time   | 1.8 seconds  |
| Supported Languages | 100+         |
| Batch Size          | 2048         |
| API Rate Limit      | 10K / minute |

---

## 🔧 Technology Stack

**Frontend**

* Streamlit
* Plotly

**Machine Learning**

* Scikit-learn
* PyTorch

**NLP**

* Cohere Embeddings
* TextBlob

**Data Processing**

* Pandas
* NumPy

---

## 🚦 API Reference

```python
from utils import get_embeddings
import cohere

co = cohere.Client(API_KEY)

embeddings = get_embeddings(
    co=co,
    texts=["Your text here"],
    model_name="multilingual-22-12"
)
```

---

## 📈 Use Cases

### 🏥 Mental Health

* Emotion tracking
* Early crisis detection
* Therapy and wellness support

### 💼 Business Intelligence

* Customer feedback analysis
* Brand monitoring
* Product sentiment insights

### 🎓 Research & Education

* Behavioral analysis
* Student engagement tracking
* NLP experimentation

### 📱 Social Media Monitoring

* Trend detection
* Audience mood tracking
* Reputation management

---

## 🤝 Contributing

We welcome contributions from the community.

1. Fork the repository
2. Create a new branch

   ```
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes

   ```
   git commit -m "Add AmazingFeature"
   ```
4. Push the branch

   ```
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.
See the `LICENSE` file for full details.

---

## 📬 Contact

Project Repository:
https://github.com/faisalshahzadnadeem/-EmoAnalyzer-Multilingual-Emotion-Sentiment-Intelligence



---

## 🙏 Acknowledgments

* Cohere — Multilingual embedding API
* Streamlit — Rapid web application framework
* Scikit-learn — Machine learning tools

---

⭐ **If you find this project useful, please consider starring the repository.**

> *Understanding human emotions through intelligent AI.*
